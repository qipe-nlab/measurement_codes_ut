

# import numpy as np
# from .base import FittingModelBase
# from measurement_tool_automation.util.util_cavity_fit import predict

# class CavitySpectralAbsolute(FittingModelBase):
#     """Fitting Model for amplitude of cavity spectral
#     """
#     # override
#     def __init__(self):
#         """Initializer of cavity spectral fitting model
#         """
#         param_names = [
#             "amplitude",
#             "decay_large",
#             "decay_small",
#             "cavity_frequency",
#         ]
#         FittingModelBase.__init__(self, param_names)
    
#     # override
#     def _model_function(self, drive_frequency, amplitude, \
#         decay_large, decay_small, cavity_frequency):
#         """Calculate amplitude of cavity respnose

#         Args:
#             drive_frequency (float or np.ndarray): drive frequency of cavity
#             amplitude (flaot): peak amplitude
#             decay_large (float): larger decay rate among external and internal decay
#             decay_small (float): small decay rate among external and internal decay
#             cavity_frequency (float): cavity resonant frequency
        
#         Returns:
#             float or np.ndarray: Amplitude of response signal
#         """
#         detune = drive_frequency - cavity_frequency
#         decay_mean = (decay_large + decay_small)/2
#         decay_diff = (decay_large - decay_small)/2
#         susceptibility = (decay_diff-1.0j*detune)/(decay_mean+1.0j*detune)
#         response = amplitude * np.abs(susceptibility)
#         return response

#     # override
#     def _initial_guess(self,x,y):
#         """Guess initial fittnig parameters

#         Args:
#             x (np.ndarray): drive frequency of cavity
#             y (np.ndarray): amplitude of cavity response
            
#         Returns:
#             dict: Fitting parameters
#         """
#         # find peak position
#         min_amplitude_index = y.argmin()
#         freq_offset = x[min_amplitude_index]

#         # find peak height and base
#         amplitude = np.percentile(y,95)
#         min_amplitude = np.min(y)
#         half_peak_height = (amplitude+min_amplitude)/2

#         # find half peak points for left and right
#         right_half_index = min_amplitude_index + (y>half_peak_height)[min_amplitude_index:].argmax()
#         left_half_index = (y<half_peak_height)[:min_amplitude_index].argmax()
#         right_half_freq = x[right_half_index]
#         left_half_freq  = x[left_half_index]
#         hwhm = (right_half_freq - left_half_freq)/2

#         # Solve equation
#         # peak_depth_ratio = (kappa_ext-kappa_int) / (kappa_ext+kappa_int) = 2 * | ( (kappa_ext-kappa_int)/2-1.j*hwhm ) / ( (kappa_ext+kappa_int)/2+1.j*hwhm )|
#         # for kappa_ext, and kappa_int.
#         # Since s11_abs is symmetric for kappa_ext and kappa_int, there is two possible solutions kappa_ext > kappa_int (over) and kappa_ext < kappa_int (under).
#         peak_depth_ratio = min_amplitude / amplitude
#         decay_large = (1+peak_depth_ratio) * np.sqrt( (3+peak_depth_ratio)/(1+3*peak_depth_ratio) ) * hwhm
#         decay_small = (1-peak_depth_ratio) * np.sqrt( (3+peak_depth_ratio)/(1+3*peak_depth_ratio) ) * hwhm

#         param_dict = {
#             'amplitude'         : amplitude,
#             'decay_large'        : decay_large,
#             'decay_small'        : decay_small,
#             'cavity_frequency'  : freq_offset,
#         }
#         return param_dict



# class CavitySpectral(FittingModelBase):
#     """Fitting model for complex response of cavity spectral
#     """
#     # override
#     def __init__(self, is_over_coupling):
#         """Initializer of cavity spectral fitting model

#         Args:
#             is_over_coupling: If true, we assume kappa_ext > kappa_int in initial guess.
#         """
#         param_names = [
#             "amplitude",
#             "phase_offset",
#             "electrical_delay",
#             "external_decay",
#             "intrinsic_decay",
#             "cavity_frequency",
#         ]
#         self.is_over_coupling = is_over_coupling
#         FittingModelBase.__init__(self, param_names, datatype = complex)
    
#     # override
#     def _model_function(self, drive_frequency, amplitude, phase_offset, electrical_delay, \
#         external_decay, intrinsic_decay, cavity_frequency):
#         """Calculate complex signal of cavity respnose

#         Args:
#             drive_frequency (float or np.ndarray): drive frequency of cavity
#             amplitude (flaot): peak amplitude
#             phase_offset (float): phase offset of signal
#             electrical_delay (float): electrical delay of signal
#             external_decay (float): cavity external decay rate
#             intrinsic_decay (float): cavity intrinsic decay rate
#             cavity_frequency (float): cavity resonant frequency
        
#         Returns:
#             np.complex or np.ndarray: Complex response
#         """
#         transmission = np.exp(1.0j*(-2.*np.pi*electrical_delay*drive_frequency+phase_offset))
#         detune = drive_frequency - cavity_frequency
#         decay_diff = 0.5*(external_decay-intrinsic_decay)
#         decay_mean = 0.5*(external_decay+intrinsic_decay)
#         susceptibility = (decay_diff-1.0j*detune)/(decay_mean+1.0j*detune)
#         y = amplitude * transmission * susceptibility
#         return y

#     def _initial_guess(self,x,y):
#         """Guess initial fittnig parameters

#         Args:
#             x (np.ndarray): drive frequency of cavity
#             y (np.ndarray): amplitude of cavity response
            
#         Returns:
#             dict: Fitting parameters
#         """

#         a0,eld,po,kex,kin,f0 = predict(x,y)

#         param_dict = {
#             'amplitude'         : a0,
#             'phase_offset'      : po,
#             'electrical_delay'  : eld,
#             'cavity_frequency'  : f0,
#             'external_decay'    : kex,
#             'intrinsic_decay'   : kin
#         }
#         return param_dict

import numpy as np
from .base import FittingModelBase
import scipy.optimize as opt

class CavitySpectralAbsolute(FittingModelBase):
    """Fitting Model for amplitude of cavity spectral
    """
    # override
    def __init__(self):
        """Initializer of cavity spectral fitting model
        """
        param_names = [
            "amplitude",
            "decay_large",
            "decay_small",
            "cavity_frequency",
        ]
        FittingModelBase.__init__(self, param_names)
    
    # override
    def _model_function(self, drive_frequency, amplitude, \
        decay_large, decay_small, cavity_frequency):
        """Calculate amplitude of cavity respnose

        Args:
            drive_frequency (float or np.ndarray): drive frequency of cavity
            amplitude (flaot): peak amplitude
            decay_large (float): larger decay rate among external and internal decay
            decay_small (float): small decay rate among external and internal decay
            cavity_frequency (float): cavity resonant frequency
        
        Returns:
            float or np.ndarray: Amplitude of response signal
        """
        detune = drive_frequency - cavity_frequency
        decay_mean = (decay_large + decay_small)/2
        decay_diff = (decay_large - decay_small)/2
        susceptibility = (decay_diff-1.0j*detune)/(decay_mean+1.0j*detune)
        response = amplitude * np.abs(susceptibility)
        return response

    # override
    def _initial_guess(self,x,y):
        """Guess initial fittnig parameters

        Args:
            x (np.ndarray): drive frequency of cavity
            y (np.ndarray): amplitude of cavity response
            
        Returns:
            dict: Fitting parameters
        """
        # find peak position
        min_amplitude_index = y.argmin()
        freq_offset = x[min_amplitude_index]

        # find peak height and base
        amplitude = np.percentile(y,95)
        min_amplitude = np.min(y)
        half_peak_height = (amplitude+min_amplitude)/2

        # find half peak points for left and right
        right_half_index = min_amplitude_index + (y>half_peak_height)[min_amplitude_index:].argmax()
        left_half_index = (y<half_peak_height)[:min_amplitude_index].argmax()
        right_half_freq = x[right_half_index]
        left_half_freq  = x[left_half_index]
        hwhm = (right_half_freq - left_half_freq)/2

        # Solve equation
        # peak_depth_ratio = (kappa_ext-kappa_int) / (kappa_ext+kappa_int) = 2 * | ( (kappa_ext-kappa_int)/2-1.j*hwhm ) / ( (kappa_ext+kappa_int)/2+1.j*hwhm )|
        # for kappa_ext, and kappa_int.
        # Since s11_abs is symmetric for kappa_ext and kappa_int, there is two possible solutions kappa_ext > kappa_int (over) and kappa_ext < kappa_int (under).
        peak_depth_ratio = min_amplitude / amplitude
        decay_large = (1+peak_depth_ratio) * np.sqrt( (3+peak_depth_ratio)/(1+3*peak_depth_ratio) ) * hwhm
        decay_small = (1-peak_depth_ratio) * np.sqrt( (3+peak_depth_ratio)/(1+3*peak_depth_ratio) ) * hwhm

        param_dict = {
            'amplitude'         : amplitude,
            'decay_large'        : decay_large,
            'decay_small'        : decay_small,
            'cavity_frequency'  : freq_offset,
        }
        return param_dict



class CavitySpectral(FittingModelBase):
    """Fitting model for complex response of cavity spectral
    """
    # override
    def __init__(self):
        """Initializer of cavity spectral fitting model

        Args:
            is_over_coupling: If true, we assume kappa_ext > kappa_int in initial guess.
        """
        param_names = [
            "amplitude",
            "phase_offset",
            "electrical_delay",
            "external_decay",
            "intrinsic_decay",
            "cavity_frequency",
        ]
        FittingModelBase.__init__(self, param_names, datatype = complex)
    
    # override
    def _model_function(self, drive_frequency, amplitude, phase_offset, electrical_delay, \
        external_decay, intrinsic_decay, cavity_frequency):
        """Calculate complex signal of cavity respnose

        Args:
            drive_frequency (float or np.ndarray): drive frequency of cavity
            amplitude (flaot): peak amplitude
            phase_offset (float): phase offset of signal
            electrical_delay (float): electrical delay of signal
            external_decay (float): cavity external decay rate
            intrinsic_decay (float): cavity intrinsic decay rate
            cavity_frequency (float): cavity resonant frequency
        
        Returns:
            np.complex or np.ndarray: Complex response
        """

        detune          = drive_frequency - cavity_frequency
        transmission    = np.exp(1.0j*(-2.*np.pi*electrical_delay*detune+phase_offset))
        decay_diff      = 0.5*(external_decay-intrinsic_decay)
        decay_mean      = 0.5*(external_decay+intrinsic_decay)
        susceptibility  = (decay_diff-1.0j*detune)/(decay_mean+1.0j*detune)
        y               = amplitude * transmission * susceptibility
        return y

    def _initial_guess(self,x,y):
        # store original_phase
        original_phase      = np.unwrap(np.angle(y))
        # remove electrical_delay
        electrical_delay    = get_electrical_delay(x,y)
        y                   = np.exp(1.0j*(2.*np.pi*electrical_delay*x))*y
        # find cavity_frequency
        index               = np.abs(np.gradient(y)).argmax()
        cavity_frequency    = x[index]
#         phase_offset        = phase_mod(original_phase[index])
        phase_offset        = abs(np.gradient(np.unwrap(np.angle(y)))).argmax()
        # fit circle
        xc,yc,rc            = fit_circle(y.real,y.imag)
        y                   -= xc + 1.0j*yc
        amplitude           = abs(xc + 1.0j*yc) + rc
        # find linewidth
        peak_phase          = np.unwrap(np.angle(y))[index]
        left_half_freq      = x[abs(y - rc*np.exp(1.0j*(peak_phase+0.5*np.pi))).argmin()]
        right_half_freq     = x[abs(y - rc*np.exp(1.0j*(peak_phase-0.5*np.pi))).argmin()]
        fwhm                = abs(right_half_freq - left_half_freq)
        external_decay      = fwhm*rc/amplitude
        intrinsic_decay     = fwhm - external_decay

        param_dict = {
            'amplitude'         : amplitude,
            'phase_offset'      : phase_offset,
            'electrical_delay'  : electrical_delay,
            'cavity_frequency'  : cavity_frequency,
            'external_decay'    : external_decay,
            'intrinsic_decay'   : intrinsic_decay
        }
        return param_dict

# def phase_mod(phase):
#     phase = np.mod(phase,2*np.pi)
#     if phase > np.pi:
#         phase -= np.pi
#     return phase
    
def get_electrical_delay(x,y):

    def evaluate(electrical_delay):
        _y       = np.exp(1.0j*(2.*np.pi*electrical_delay*x))*y
        xc,yc,rc = fit_circle(_y.real, _y.imag)
        return abs((rc**2 - ((_y.real-xc)**2 + (_y.imag-yc)**2)).sum())

    phase       = np.unwrap(np.angle(y))
    # eld_under   = abs((phase[-1]-phase[0])/(2*np.pi*(x[-1]-x[0])))
    # eld_over    = abs((abs(phase[-1]-phase[0]) - 2*np.pi)/(2*np.pi*(x[-1]-x[0])))
    eld_under   = (phase[-1]-phase[0])/(2*np.pi*(x[-1]-x[0]))
    eld_over    = (abs(phase[-1]-phase[0]) - 2*np.pi)/(2*np.pi*(x[-1]-x[0]))
    cost_under  = evaluate(eld_under)
    cost_over   = evaluate(eld_over)
    if cost_under > cost_over:
        electrical_delay = eld_over
    else:
        electrical_delay = eld_under
    return electrical_delay

def fit_circle(x,y):
    z   = x**2 + y**2
    mn  = x.size
    mx  = x.sum()
    my  = y.sum()
    mz  = z.sum()
    mxx = (x*x).sum()
    myy = (y*y).sum()
    mzz = (z*z).sum()
    mxy = (x*y).sum()
    myz = (y*z).sum()
    mzx = (z*x).sum()
    M   = np.array([
        [mzz, mzx, myz, mz],
        [mzx, mxx, mxy, mx],
        [myz, mxy, myy, my],
        [ mz,  mx,  my, mn]
    ])
    B   = np.array([
        [ 0, 0, 0, -2],
        [ 0, 1, 0,  0],
        [ 0, 0, 1,  0],
        [-2, 0, 0,  0]
    ])

    def cost(eta):
        return np.linalg.det(M-eta*B)**2
    
    res = opt.minimize(cost,x0=0,method="BFGS")
    eig = np.linalg.eig(M-res.x*B)
    A   = eig[1].T[abs(eig[0]).argmin()]
    A   /= (A@B@A)**0.5
    xc  = - 0.5*A[1]/A[0]
    yc  = - 0.5*A[2]/A[0]
    rc  = 0.5/abs(A[0])
    return xc,yc,rc




