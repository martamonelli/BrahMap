import gc
from typing import List, Union, Optional
from dataclasses import dataclass, asdict

import numpy as np
import litebird_sim as lbs

from ..core import GLSParameters, GLSResult, compute_GLS_maps_from_PTS, DTypeNoiseCov

from ..lbsim import LBSimProcessTimeSamples, DTypeLBSNoiseCov

from ..math import DTypeFloat

import scipy as sp
from scipy.sparse.linalg import cg, LinearOperator ### MM: added
from scipy.interpolate import CubicSpline          ### MM: added
from scipy.linalg import lstsq

from time import time

@dataclass
class LBSimGLSParameters(GLSParameters):
    return_processed_samples: bool = False
    output_coordinate_system: lbs.CoordinateSystem = lbs.CoordinateSystem.Galactic


@dataclass
class LBSimGLSResult(GLSResult):
    nside: int
    coordinate_system: lbs.CoordinateSystem
    
    
####################################################
# DEFINE FUCTIONS FOR INPAINTING
####################################################

def P_oof_inv_func(N, sampling_rate_hz, net_ukrts, fknee_mhz, alpha, fmin_hz):
    '''
    Given N, the sampling rate and all 1/f noise parameters, returns P^-1
    '''
    sigma = net_ukrts * np.sqrt(sampling_rate_hz) / 1e6     # as in LBS rescale_noise
    freqs = sp.fft.rfftfreq(N, d=1/sampling_rate_hz)
    P_oof_inv = 1/(sigma**2*(freqs**alpha + (fknee_mhz*1e-3)**alpha)/(freqs**alpha + fmin_hz**alpha)*len(freqs))
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
    z_fft = sp.fft.rfft(z)
    product = Pinv * z_fft
    result = sp.fft.irfft(product)
    return result[:N-n]

####################################################
# BACK TO BRAHMAP
####################################################

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

                # total length of the inpainted TOD
                nsamp_inpainted = 2*nsamp_temp

                # inverse of the 1/f power spectra
                P_oof_inv = P_oof_inv_func(nsamp_inpainted, sampling_rate_hz, net_ukrts, fknee_mhz, alpha, fmin_hz)
                
                nn = 100

                tod_temp_binned = np.empty(int(nsamp_temp/nn))

                for i in range(len(tod_temp_binned)):
                    tod_temp_binned[i] = np.mean(tod_temp[i*nn:(i+1)*nn])

                nsamp_binned = len(tod_temp_binned)
                nsamp_inpainted_binned = 2*nsamp_binned

                nyquist_binned = sampling_rate_hz/2/nn

                freqs = sp.fft.rfftfreq(nsamp_inpainted, d=1/sampling_rate_hz)
                mask_freqs = np.where(freqs<=nyquist_binned)

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
                    z_fft = sp.fft.rfft(z)
                    product = P_oof_inv_binned * z_fft
                    result = sp.fft.irfft(product)
                    return result[:lenx_binned]

                # Define the LinearOperator for CG
                A_op_binned = LinearOperator((lenx_binned,lenx_binned), matvec=A_func_x_only_binned)

                x_sol_10_binned, info = cg(A_op_binned, b_binned, rtol=1e-10)

                x = nn*(1/2 + np.arange(nsamp_binned))
                y = x_sol_10_binned
                cs = CubicSpline(x, y)

                x_sol_10_binned_spline = cs(np.arange(nsamp_temp))

                time_ordered_data[start_idx:end_idx] = x_sol_10_binned_spline
                
                start_idx = end_idx
                         
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
