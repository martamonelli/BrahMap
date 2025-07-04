import gc
from typing import List, Union, Optional
from dataclasses import dataclass, asdict

import numpy as np
import litebird_sim as lbs

from ..core import GLSParameters, GLSResult, compute_GLS_maps_from_PTS, DTypeNoiseCov

from ..lbsim import LBSimProcessTimeSamples, DTypeLBSNoiseCov

from ..math import DTypeFloat

from scipy.sparse.linalg import cg, LinearOperator ### MM: added
from time import time

@dataclass
class LBSimGLSParameters(GLSParameters):
    return_processed_samples: bool = False
    output_coordinate_system: lbs.CoordinateSystem = lbs.CoordinateSystem.Galactic


@dataclass
class LBSimGLSResult(GLSResult):
    nside: int
    coordinate_system: lbs.CoordinateSystem
    
    
def check_lr(left,right):
    '''
    Checks if the arguments left and right are well defined (they must be different)
    '''
    if left == right:
        print('one (and only one) argument between left and right must be True!')
        quit()


def check_Pv(P,v):
    '''
    Checks if the arguments P and v are well defined (len(v) must be len(P)-1)
    '''
    if len(v) != len(P)-1:
        print('len(v) must be len(P)-1!')
        quit()


def padding(v,N,left=False,right=False):
    '''
    Given v, returns a padded vector, (0,v) or (v,0) depending on whether left or right are True.
    
    ARGUMENTS____________________________________________ 
    v:     vector to be padded
    N:     final length of the padded vector
    left:  if true, returns (0,v) 
    right: if true, returns (v,0) 
    '''
    check_lr(left,right)
    n = len(v)
    z = np.zeros(N)
    if right == True:
        z[:n] = v
    elif left == True:
        z[-n:] = v    
    return z


def A_func(P,v,left=False,right=False):    
    # z = padded v
    # A(z) = IDFT(1/P * DFT(z))
    check_lr(left,right)
    check_Pv(P,v)
    n = len(v)
    N = 2*n
    z = padding(v,N,left,right)
    z_fft = np.fft.rfft(z)
    product = 1/P * z_fft
    result = np.fft.irfft(product).real
    return result[:n]


def LBSim_compute_GLS_maps(
    nside: int,
    observations: Union[lbs.Observation, List[lbs.Observation]],
    pointings: Union[np.ndarray, List[np.ndarray], None] = None,
    hwp: Optional[lbs.HWP] = None,
    components: Union[str, List[str]] = "tod",
    pointings_flag: Union[np.ndarray, None] = None,
    inv_noise_cov_operator: Union[DTypeNoiseCov, DTypeLBSNoiseCov, None] = None,
    threshold: float = 1.0e-5,
    dtype_float: Union[DTypeFloat, None] = None,
    LBSim_gls_parameters: LBSimGLSParameters = LBSimGLSParameters(),
    inpainting: bool = False,
) -> Union[LBSimGLSResult, tuple[LBSimProcessTimeSamples, LBSimGLSResult]]:
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
            obs_list=observations,
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
            
            sigmas = nets_ukrts * np.sqrt(sampling_rate_hz) / 1e6 #as in rescale_noise
            
            start = time()

            for det_idx in range(obs.n_detectors):
                
                tod_temp = getattr(obs, components[0])[det_idx]
                nsamp_temp = len(tod_temp)
                
                fknee_mhz = fknees_mhz[det_idx]
                fmin_hz = fmins_hz[det_idx]
                alpha = alphas[det_idx]
                net_ukrts = nets_ukrts[det_idx]
                sigma = sigmas[det_idx]
                
                end_idx += obs.n_samples
                time_ordered_data[start_idx:end_idx] = tod_temp
                
                start_idx = end_idx
                end_idx += obs.n_samples

                freqs = np.fft.rfftfreq(2*nsamp_temp, d=1/sampling_rate_hz) #2*nsamp_temp!
                P_oof = sigma**2*(freqs**alpha + (fknee_mhz*1e-3)**alpha)/(freqs**alpha + fmin_hz**alpha)*len(freqs)
                
                b = -A_func(P_oof, tod_temp, left=True)	#A_func applied to a vector [0,y]
                
                def A_func_x_only(x):
                    return A_func(P_oof, x, right=True)	#A_func applied to a vector [x,0]
               
                # Define the LinearOperator for CG
                A_op = LinearOperator((nsamp_temp,nsamp_temp), matvec=A_func_x_only)

                # Initial guess for x
                avg_head = np.mean(tod_temp[:10])
                avg_tail = np.mean(tod_temp[-10:])
                x0 = avg_tail + np.arange(nsamp_temp)/nsamp_temp*(avg_head-avg_tail)

                x_sol, info = cg(A_op, b, x0=x0, rtol=1e-10)
                
                time_ordered_data[start_idx:end_idx] = x_sol
                
                start_idx = end_idx

            print(time()-start)
                         
    else: 
        time_ordered_data = np.concatenate(
            [getattr(obs, components[0]) for obs in observations], axis=None
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
