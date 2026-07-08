import gc
from typing import List
from dataclasses import dataclass, asdict
import numpy as np
import numpy.typing as npt
import litebird_sim as lbs

from ..base import DTypeNoiseCov

from ..core import GLSParameters, GLSResult, compute_GLS_maps_from_PTS

from ..lbsim import LBSimProcessTimeSamples, LBSimProcessTimeSamplesInpainting, LBSimProcessTimeSamplesZeroPadding, DTypeLBSNoiseCov

from ..math import DTypeFloat

from ..lbsim.utils_inpainting import inpainting_func

import time


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
    x0: Union[np.ndarray, None] = None,
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
    x0 : np.ndarray, optional
        Initial guess for the GLS solution in the form 
        [I_1, Q_1, U_1, I_2, Q_2, U_2, ...], by default None

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

    time_ordered_data = np.concatenate(
        [getattr(obs, components[0]) for obs in processed_samples.obs_list], axis=None
    )

    gls_result = compute_GLS_maps_from_PTS(
        processed_samples=processed_samples,
        time_ordered_data=time_ordered_data,
        inv_noise_cov_operator=inv_noise_cov_operator,
        gls_parameters=LBSim_gls_parameters,
        x0=x0,
    )

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


def LBSim_compute_GLS_maps_inpainting(
    nside: int,
    observations: Union[lbs.Observation, List[lbs.Observation]],
    inpainting_len: int,
    samples_per_bin: int,
    trash_pix_per_chunk: int,
    pointings: Union[np.ndarray, List[np.ndarray], None] = None,
    hwp: Optional[lbs.HWP] = None,
    components: Union[str, List[str]] = "tod",
    pointings_flag: Optional[np.ndarray] = None,
    inv_noise_cov_operator: Union[DTypeNoiseCov, DTypeLBSNoiseCov, None] = None,
    threshold: float = 1.0e-5,
    dtype_float: Optional[DTypeFloat] = None,
    LBSim_gls_parameters: LBSimGLSParameters = LBSimGLSParameters(),
    x0: Union[np.ndarray, None] = None,
) -> Union[LBSimGLSResult, tuple[LBSimProcessTimeSamples, LBSimGLSResult]]:
    """_summary_

    Parameters
    ----------
    nside : int
        _description_
    observations : Union[lbs.Observation, List[lbs.Observation]]
        _description_
    inpainting_len : int
        The number of inpainted samples per chunk
    samples_per_bin : int
        Number of samples per bin for the inpaining algorithm
    trash_pix_per_chunk : int
        Number of trash pixels where to project the inpainted samples per chunk
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
    x0 : np.ndarray, optional
        Initial guess for the GLS solution in the form 
        [I_1, Q_1, U_1, I_2, Q_2, U_2, ...], by default None

    Returns
    -------
    Union[LBSimGLSResult, tuple[LBSimProcessTimeSamples, LBSimGLSResult]]
        _description_
    """

    assert all(x > 0 for x in [inpainting_len, samples_per_bin, trash_pix_per_chunk]), "all inpainting parameters must be positive"

    if inv_noise_cov_operator is None:
        noise_weights = None
    else:
        noise_weights = inv_noise_cov_operator.diag

    processed_samples = LBSimProcessTimeSamplesInpainting(
        nside=nside,
        observations=observations,
        inpainting_len=inpainting_len,
        trash_pix_per_chunk=trash_pix_per_chunk,
        pointings=pointings,
        hwp=hwp,
        pointings_flag=pointings_flag,
        solver_type=LBSim_gls_parameters.solver_type,
        noise_weights=noise_weights,
        output_coordinate_system=LBSim_gls_parameters.output_coordinate_system,
        threshold=threshold,
        dtype_float=dtype_float,
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

    start = time.time()

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

            fknee_hz = fknees_mhz[det_idx]*1e-3
            fmin_hz = fmins_hz[det_idx]
            alpha = alphas[det_idx]
            net_ukrts = nets_ukrts[det_idx]

            start_idx = end_idx
            end_idx += nsamp_temp
            time_ordered_data[start_idx:end_idx] = tod_temp

            start_idx = end_idx
            end_idx += inpainting_len

            x_sol = inpainting_func(tod_temp, inpainting_len, net_ukrts, fknee_hz, alpha, fmin_hz, sampling_rate_hz, samples_per_bin)

            time_ordered_data[start_idx:end_idx] = x_sol

            start_idx = end_idx
    
    print(f"preprocessing took {time.time()-start} seconds")

    start = time.time()

    gls_result = compute_GLS_maps_from_PTS(
        processed_samples=processed_samples,
        time_ordered_data=time_ordered_data,
        inv_noise_cov_operator=inv_noise_cov_operator,
        gls_parameters=LBSim_gls_parameters,
        x0=x0,
    )

    print(f"map-making took {time.time()-start} seconds")

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


def LBSim_compute_GLS_maps_zero_padding(
    nside: int,
    observations: Union[lbs.Observation, List[lbs.Observation]],
    zero_padding_len: int,
    pointings: Union[np.ndarray, List[np.ndarray], None] = None,
    hwp: Optional[lbs.HWP] = None,
    components: Union[str, List[str]] = "tod",
    pointings_flag: Optional[np.ndarray] = None,
    inv_noise_cov_operator: Union[DTypeNoiseCov, DTypeLBSNoiseCov, None] = None,
    threshold: float = 1.0e-5,
    dtype_float: Optional[DTypeFloat] = None,
    LBSim_gls_parameters: LBSimGLSParameters = LBSimGLSParameters(),
    x0: Union[np.ndarray, None] = None,
) -> Union[LBSimGLSResult, tuple[LBSimProcessTimeSamples, LBSimGLSResult]]:
    """_summary_

    Parameters
    ----------
    nside : int
        _description_
    observations : Union[lbs.Observation, List[lbs.Observation]]
        _description_
    zero_padding_len : int
        The number of zeros padded per chunk
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
    x0 : np.ndarray, optional
        Initial guess for the GLS solution in the form 
        [I_1, Q_1, U_1, I_2, Q_2, U_2, ...], by default None

    Returns
    -------
    Union[LBSimGLSResult, tuple[LBSimProcessTimeSamples, LBSimGLSResult]]
        _description_
    """

    assert zero_padding_len > 0, "zero_padding_len must be positive"

    if inv_noise_cov_operator is None:
        noise_weights = None
    else:
        noise_weights = inv_noise_cov_operator.diag

    processed_samples = LBSimProcessTimeSamplesZeroPadding(
        nside=nside,
        observations=observations,
        zero_padding_len=zero_padding_len,
        pointings=pointings,
        hwp=hwp,
        pointings_flag=pointings_flag,
        solver_type=LBSim_gls_parameters.solver_type,
        noise_weights=noise_weights,
        output_coordinate_system=LBSim_gls_parameters.output_coordinate_system,
        threshold=threshold,
        dtype_float=dtype_float,
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

            start_idx = end_idx
            end_idx += nsamp_temp
            time_ordered_data[start_idx:end_idx] = tod_temp

            start_idx = end_idx
            end_idx += zero_padding_len

            time_ordered_data[start_idx:end_idx] = 0

            start_idx = end_idx

    gls_result = compute_GLS_maps_from_PTS(
        processed_samples=processed_samples,
        time_ordered_data=time_ordered_data,
        inv_noise_cov_operator=inv_noise_cov_operator,
        gls_parameters=LBSim_gls_parameters,
        x0=x0,
    )

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