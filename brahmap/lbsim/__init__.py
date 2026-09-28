from typing import Union

from .lbsim_process_time_samples import (
    LBSimProcessTimeSamples,
    LBSimProcessTimeSamplesInpainting,
    LBSimProcessTimeSamplesZeroPadding,
)

from .lbsim_noise_operators import (
    LBSim_InvNoiseCovLO_UnCorr,
    LBSim_InvNoiseCovLO_Circulant,
    LBSim_InvNoiseCovLO_Circulant_ExtraSamples,
    LBSim_InvNoiseCovLO_Toeplitz,
)

DTypeLBSNoiseCov = Union[
    LBSim_InvNoiseCovLO_UnCorr,
    LBSim_InvNoiseCovLO_Circulant,
    LBSim_InvNoiseCovLO_Toeplitz,
]

from .lbsim_GLS import (
    LBSimGLSParameters, 
    LBSimGLSResult, 
    LBSim_compute_GLS_maps,   # noqa: E402
    LBSim_compute_GLS_maps_inpainting,
    LBSim_compute_GLS_maps_zero_padding
)

__all__ = [
    # lbsim_process_time_samples.py
    "LBSimProcessTimeSamples",
    "LBSimProcessTimeSamplesInpainting",
    "LBSimProcessTimeSamplesZeroPadding",
    # lbsim_noise_operators.py
    "LBSim_InvNoiseCovLO_UnCorr",
    "LBSim_InvNoiseCovLO_Circulant"
    "LBSim_InvNoiseCovLO_Circulant_ExtraSamples",
    "LBSim_InvNoiseCovLO_Toeplitz",
    # lbsim_GLS.py
    "LBSimGLSParameters",
    "LBSimGLSResult",
    "LBSim_compute_GLS_maps",
    "LBSim_compute_GLS_maps_inpainting",
    "LBSim_compute_GLS_maps_zero_padding",
    # NoiseCov dtype
    "DTypeLBSNoiseCov",
]
