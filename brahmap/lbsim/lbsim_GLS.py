import gc
from typing import List
from dataclasses import dataclass, asdict
import numpy as np
import numpy.typing as npt
import litebird_sim as lbs

from ..base import DTypeNoiseCov

from ..core import GLSParameters, GLSResult, compute_GLS_maps_from_PTS

from ..lbsim import LBSimProcessTimeSamples, DTypeLBSNoiseCov

from ..math import DTypeFloat

import scipy as sp
from scipy.sparse.linalg import cg, LinearOperator
from scipy.interpolate import CubicSpline

@dataclass
class LBSimGLSParameters(GLSParameters):
    """A data class encapsulating the configuration parameters for the
    Generalized Least Squares (GLS) map-making algorithm with `litebird_sim` data.

    Attributes
    ----------
    solver_type : SolverType
        The map-making solver configuration to use (e.g. $I$, $QU$, $IQU$)
    use_iterative_solver : bool
        Whether to enforce the use of an iterative solver (like PCG) for map-making
    isolver_threshold : float
        The numerical tolerance threshold for the iterative solver to
        declare convergence
    isolver_max_iterations : int
        The maximum number of iterations allowed for the iterative solver
    callback_function : Callable
        A callable function executed at each iteration of the solver
    return_processed_samples : bool
        Whether the GLS solver function should return the processed time
        samples container
    return_hit_map : bool
        Whether the function should return the pixel hit map
    output_coordinate_system : lbs.CoordinateSystem
        The celestial coordinate system to use for the generated output maps
    """

    return_processed_samples: bool = False
    output_coordinate_system: lbs.CoordinateSystem = lbs.CoordinateSystem.Galactic


@dataclass
class LBSimGLSResult(GLSResult):
    """A data class storing the output results of the GLS map-making done with
    `litebird_sim` data.

    Attributes
    ----------
    solver_type : SolverType
        The map-making solver configuration (e.g. $I$, $QU$, $IQU$)
    npix : int
        The number of pixels in the sky map
    new_npix : int
        The number of valid pixels actually observed and processed
    GLS_maps : npt.NDArray[np.number]
        The final Generalized Least Squares (GLS) estimated sky maps
    hit_map : npt.NDArray[np.number] | None
        The array representing the total number of hits per pixel
    convergence_status : bool
        A boolean indicating whether the iterative solver successfully converged
    num_iterations : int
        The total number of iterations actually performed by the solver before stopping
    GLSParameters : LBSimGLSParameters
        The input parameters configuration used for the GLS map-making
    nside : int
        The HEALPix resolution parameter defining the number of pixels
    coordinate_system : lbs.CoordinateSystem
        The coordinate system to use for map-making (e.g., Galactic, Ecliptic)
    GLSParameters : GLSParameters
        The input parameters configuration used for the GLS map-making
    """

    nside: int
    coordinate_system: lbs.CoordinateSystem

####################################################
# DEFINE FUCTIONS FOR INPAINTING

def P_oof_inv_func(N, sampling_rate_hz, net_ukrts, fknee_mhz, alpha, fmin_hz):
    '''
    Given N, the sampling rate and all 1/f noise parameters, returns P^-1
    '''
    sigma = net_ukrts * np.sqrt(sampling_rate_hz) / 1e6     # as in LBS rescale_noise
    abs_freqs = np.abs(np.fft.fftfreq(N, d=1/sampling_rate_hz))
    P_oof_inv = 1/(sigma**2*(abs_freqs**alpha + (fknee_mhz*1e-3)**alpha)/(abs_freqs**alpha + fmin_hz**alpha)*len(abs_freqs))
    return P_oof_inv

def A_func_left(Pinv, y, N):   
    '''
    Given y, pads it to the left and returns the first len(y) elements of IDFT(1/P * DFT[0,y]))
    
    ARGUMENTS____________________________________________ 
    Pinv:  inverse of the power spectrum
    y:     vector to be padded
    ''' 
    n = len(y)
    z = np.concatenate((np.zeros(N-n), y))
    z_fft = np.fft.fft(z)
    product = Pinv * z_fft
    result = np.fft.ifft(product)
    return result[:N-n]

####################################################

def LBSim_compute_GLS_maps(
    nside: int,
    observations: lbs.Observation | List[lbs.Observation],
    pointings: npt.NDArray[np.number] | List[npt.NDArray[np.number]] | None = None,
    hwp: lbs.HWP | None = None,
    components: str | List[str] = "tod",
    pointings_flag: npt.NDArray[np.bool_] | None = None,
    inv_noise_cov_operator: DTypeNoiseCov | DTypeLBSNoiseCov | None = None,
    threshold: float = 1.0e-5,
    dtype_float: DTypeFloat = np.float64,
    LBSim_gls_parameters: LBSimGLSParameters = LBSimGLSParameters(),
    x0: npt.NDArray[np.number] | None = None,
    inpainting: bool = False,
) -> LBSimGLSResult | tuple[LBSimProcessTimeSamples, LBSimGLSResult]:
    """Computes the Generalized Least Squares (GLS) maps from
    `litebird_sim` observations.

    Parameters
    ----------
    nside : int
        The HEALPix $N_{side}$ resolution parameter defining the number of pixels
    observations : lbs.Observation | List[lbs.Observation]
        An instance of the `Observation` class or a list of the same
    pointings : npt.NDArray[np.number] | List[npt.NDArray[np.number]] | None, optional
        Array of detector pointing indices mapping time samples to observed sky pixels,
        by default `None`
    hwp : lbs.HWP | None, optional
        The Half-Wave Plate (HWP) angles or configuration, by default `None`
    components : str | List[str], optional
        A string or list defining the TOD components to be used for map-making, by
        default `"tod"`
    pointings_flag : npt.NDArray[np.bool_] | None, optional
        Boolean array indicating valid pointing samples, by default `None`.
        The `True` value indicates a valid pointing, and the `False`
        value indicates a bad pointing. If set as `None`, all the
        pointings are considered valid
    inv_noise_cov_operator : DTypeNoiseCov | DTypeLBSNoiseCov | None, optional
        The inverse noise covariance linear operator ($N^{-1}$), by default `None`
    threshold : float, optional
        The condition number threshold used to flag degenerate or
        under-sampled pixels, by default `1.0e-5`
    dtype_float : DTypeFloat, optional
        The data type to use for floating point arrays, by default
        `np.float64`
    LBSim_gls_parameters : LBSimGLSParameters, optional
        The parameter configuration dictating the map-making behavior, by
        default `LBSimGLSParameters()`
    x0 : npt.NDArray[np.number] | None, optional
        Initial guess for the GLS solution in the form of interleaved
        maps (e.g. $[I_1, Q_1, U_1, I_2, Q_2, U_2, \\dots]$), by default `None`

    Returns
    -------
    LBSimGLSResult | tuple[LBSimProcessTimeSamples, LBSimGLSResult]
        The dataclass containing the final output from the GLS map-maker,
        optionally returning the processed samples container
    """
    if inv_noise_cov_operator is None:
        noise_weights = None
    else:
        noise_weights = inv_noise_cov_operator.diag

    processed_samples = LBSimProcessTimeSamples(
        nside=nside,
        observations=observations,
        pointings=pointings,
        hwp=hwp,
        pointings_flag=pointings_flag,
        solver_type=LBSim_gls_parameters.solver_type,
        noise_weights=noise_weights,
        output_coordinate_system=LBSim_gls_parameters.output_coordinate_system,
        threshold=threshold,
        dtype_float=dtype_float,
        inpainting=inpainting,
    )

    if isinstance(components, str):
        components = [components]

    if len(components) > 1:
        lbs.mapmaking.destriper._sum_components_into_obs(
            obs_list=processed_samples.obs_list,
            target=components[0],
            other_components=components[1:],
            factor=1.0,
        )

    if inpainting:
        time_ordered_data = np.empty(processed_samples.nsamples)

        end_idx = 0

        for obs in observations:
            fknees_mhz = obs.fknee_mhz
            fmins_hz = obs.fmin_hz
            alphas = obs.alpha
            nets_ukrts = obs.net_ukrts
            sampling_rate_hz = obs.sampling_rate_hz

            for det_idx in range(obs.n_detectors):   
                tod_temp = obs.tod[det_idx]
                nsamp_temp = len(tod_temp)

                fknee_mhz = fknees_mhz[det_idx]
                fmin_hz = fmins_hz[det_idx]
                alpha = alphas[det_idx]
                net_ukrts = nets_ukrts[det_idx]

                start_idx = end_idx
                end_idx += nsamp_temp
                time_ordered_data[start_idx:end_idx] = tod_temp

                start_idx = end_idx
                end_idx += nsamp_temp

                # total length of the inpainted TOD
                nsamp_inpainted = 2*nsamp_temp

                # inverse of the 1/f power spectra
                P_oof_inv = P_oof_inv_func(nsamp_inpainted, sampling_rate_hz, net_ukrts, fknee_mhz, alpha, fmin_hz)

                nn = 8 #FIXME: by hand, but smaller than detector_sampling_freq/(fknee_mhz*1e-3) = 5

                tod_temp_binned = np.empty(int(nsamp_temp/nn))

                for i in range(len(tod_temp_binned)):
                    tod_temp_binned[i] = np.mean(tod_temp[i*nn:(i+1)*nn])

                nsamp_binned = len(tod_temp_binned)
                nsamp_inpainted_binned = 2*nsamp_binned

                nyquist_binned = sampling_rate_hz/2/nn

                freqs = np.fft.fftfreq(nsamp_inpainted, d=1/sampling_rate_hz)
                mask_freqs = np.where((freqs<nyquist_binned) & (freqs>=-nyquist_binned))

                # inverse of the 1/f power spectra
                P_oof_inv_binned = P_oof_inv[mask_freqs]*nn

                # -IDFT(1/P * DFT([0,y]))
                b_binned = -A_func_left(P_oof_inv_binned, tod_temp_binned, nsamp_inpainted_binned)

                lenx_binned = nsamp_inpainted_binned - nsamp_binned

                # we need a function of x only to build the LinearOperator for CG
                def A_func_x_only_binned(x):   
                    # Given x, computes A_func(P_oof_inv, x, nsamp_inpainted, right=True)
                    
                    z = np.concatenate((x, np.zeros(nsamp_binned)))
                    z_fft = np.fft.fft(z)
                    product = P_oof_inv_binned * z_fft
                    result = np.fft.ifft(product)
                    return result[:lenx_binned]

                # Define the LinearOperator for CG
                A_op_binned = LinearOperator((lenx_binned,lenx_binned), matvec=A_func_x_only_binned)

                x_sol_10_binned, info = cg(A_op_binned, b_binned, rtol=1e-5)

                x = nn*(1/2 + np.arange(nsamp_binned))
                y = x_sol_10_binned
                cs = CubicSpline(x, y)

                x_sol_10_binned_spline = cs(np.arange(nsamp_temp))

                time_ordered_data[start_idx:end_idx] = x_sol_10_binned_spline

                start_idx = end_idx
    else: 
        time_ordered_data = np.concatenate(
            [getattr(obs, components[0]) for obs in processed_samples.obs_list], axis=None
        )

    lbsim_gls_result = compute_GLS_maps_from_PTS(
        processed_samples=processed_samples,
        time_ordered_data=time_ordered_data,
        inv_noise_cov_operator=inv_noise_cov_operator,
        gls_parameters=LBSim_gls_parameters,
        x0=x0,
    )

    if inpainting:
        lbsim_gls_result.GLS_maps = lbsim_gls_result.GLS_maps[:,:12*nside**2]

    lbsim_gls_result = LBSimGLSResult(
        nside=nside,
        coordinate_system=LBSim_gls_parameters.output_coordinate_system,
        **asdict(gls_result),
    )

    if LBSim_gls_parameters.return_processed_samples:
        return processed_samples, lbsim_gls_result
    else:
        del processed_samples
        gc.collect()
        return lbsim_gls_result
