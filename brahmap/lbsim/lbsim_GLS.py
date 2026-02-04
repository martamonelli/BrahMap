import gc
from typing import List, Union, Optional
from dataclasses import dataclass, asdict

import numpy as np
import litebird_sim as lbs

from ..core import GLSParameters, GLSResult, compute_GLS_maps_from_PTS, DTypeNoiseCov

from ..lbsim import LBSimProcessTimeSamples, DTypeLBSNoiseCov

from ..math import DTypeFloat

import scipy as sp
from scipy.sparse.linalg import cg, LinearOperator
from scipy.interpolate import CubicSpline

@dataclass
class LBSimGLSParameters(GLSParameters):
    """A class to encapsulate the parameters used for GLS map-making with
    `litebird_sim` data

    Parameters
    ----------
    solver_type : SolverType
        _description_
    use_iterative_solver : bool
        _description_
    isolver_threshold : float
        _description_
    isolver_max_iterations : int
        _description_
    callback_function : Callable
        _description_
    return_processed_samples : bool
        _description_
    return_hit_map : bool
        _description_
    return_processed_samples : bool
        _description_
    output_coordinate_system : lbs.CoordinateSystem
        _description_
    """

    return_processed_samples: bool = False
    output_coordinate_system: lbs.CoordinateSystem = lbs.CoordinateSystem.Galactic


@dataclass
class LBSimGLSResult(GLSResult):
    """A class to store the results of the GLs map-making done with `litebird_sim` data

    Parameters
    ----------
    solver_type : SolverType
        _description_
    npix : int
        _description_
    new_npix : int
        _description_
    GLS_maps : np.ndarray
        _description_
    hit_map : np.ndarray
        _description_
    convergence_status : bool
        _description_
    num_iterations : int
        _description_
    GLSParameters : GLSParameters
        _description_
    nside : int
        _description_
    coordinate_system : lbs.CoordinateSystem
        _description_
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
    observations: Union[lbs.Observation, List[lbs.Observation]],
    pointings: Union[np.ndarray, List[np.ndarray], None] = None,
    hwp: Optional[lbs.HWP] = None,
    components: Union[str, List[str]] = "tod",
    pointings_flag: Optional[np.ndarray] = None,
    inv_noise_cov_operator: Union[DTypeNoiseCov, DTypeLBSNoiseCov, None] = None,
    threshold: float = 1.0e-5,
    dtype_float: Optional[DTypeFloat] = None,
    LBSim_gls_parameters: LBSimGLSParameters = LBSimGLSParameters(),
    inpainting: bool = False,
    zeros: bool = False,
) -> Union[LBSimGLSResult, tuple[LBSimProcessTimeSamples, LBSimGLSResult]]:
    """_summary_

    Parameters
    ----------
    nside : int
        _description_
    observations : Union[lbs.Observation, List[lbs.Observation]]
        _description_
    pointings : Union[np.ndarray, List[np.ndarray], None], optional
        _description_, by default None
    hwp : Optional[lbs.HWP], optional
        _description_, by default None
    components : Union[str, List[str]], optional
        _description_, by default "tod"
    pointings_flag : Optional[np.ndarray], optional
        _description_, by default None
    inv_noise_cov_operator : Union[DTypeNoiseCov, DTypeLBSNoiseCov, None], optional
        _description_, by default None
    threshold : float, optional
        _description_, by default 1.0e-5
    dtype_float : Optional[DTypeFloat], optional
        _description_, by default None
    LBSim_gls_parameters : LBSimGLSParameters, optional
        _description_, by default LBSimGLSParameters()

    Returns
    -------
    Union[LBSimGLSResult, tuple[LBSimProcessTimeSamples, LBSimGLSResult]]
        _description_
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
        zeros=zeros,
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
        
        start_idx = 0
        end_idx = 0
        
        for obs in observations:
            fknees_mhz = obs.fknee_mhz
            fmins_hz = obs.fmin_hz
            alphas = obs.alpha
            nets_ukrts = obs.net_ukrts
            sampling_rate_hz = obs.sampling_rate_hz

            for det_idx in range(obs.n_detectors):   
                tod_temp = getattr(obs, components[0])[det_idx]
                nsamp_temp = len(tod_temp)
                
                fknee_mhz = fknees_mhz[det_idx]
                fmin_hz = fmins_hz[det_idx]
                alpha = alphas[det_idx]
                net_ukrts = nets_ukrts[det_idx]
                
                end_idx += obs.n_samples
                time_ordered_data[start_idx:end_idx] = tod_temp
                
                start_idx = end_idx
                end_idx += obs.n_samples

                if zeros:
                    time_ordered_data[start_idx:end_idx] = np.zeros(end_idx-start_idx)
                else:
                    # total length of the inpainted TOD
                    nsamp_inpainted = 2*nsamp_temp

                    # inverse of the 1/f power spectra
                    P_oof_inv = P_oof_inv_func(nsamp_inpainted, sampling_rate_hz, net_ukrts, fknee_mhz, alpha, fmin_hz)
                    
                    nn = 32 #FIXME: how should we pick this?

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
                        '''
                        Given x, computes A_func(P_oof_inv, x, nsamp_inpainted, right=True)
                        ''' 
                        z = np.concatenate((x, np.zeros(nsamp_binned)))
                        z_fft = np.fft.fft(z)
                        product = P_oof_inv_binned * z_fft
                        result = np.fft.ifft(product)
                        return result[:lenx_binned]

                    # Define the LinearOperator for CG
                    A_op_binned = LinearOperator((lenx_binned,lenx_binned), matvec=A_func_x_only_binned)

                    x_sol_10_binned, info = cg(A_op_binned, b_binned, rtol=1e-15)

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

    gls_result = compute_GLS_maps_from_PTS(
        processed_samples=processed_samples,
        time_ordered_data=time_ordered_data,
        inv_noise_cov_operator=inv_noise_cov_operator,
        gls_parameters=LBSim_gls_parameters,
    )
    
    if inpainting:
        gls_result.GLS_maps = gls_result.GLS_maps[:,:12*nside**2]

    gls_result = LBSimGLSResult(
        nside=nside,
        coordinate_system=LBSim_gls_parameters.output_coordinate_system,
        **asdict(gls_result),
    )

    if LBSim_gls_parameters.return_processed_samples:
        return processed_samples, gls_result
    else:
        del processed_samples
        gc.collect()
        return gls_result
