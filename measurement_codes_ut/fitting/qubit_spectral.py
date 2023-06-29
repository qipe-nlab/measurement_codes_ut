

import numpy as np
from .base import FittingModelBase


class QubitSpectral(FittingModelBase):
    """Fitting model for qubit Lorentzian spectral
    """
    # override
    def __init__(self):
        """Initializer of qubit spectral fitting model
        """
        param_names = [
            'amplitude',
            'qubit_frequency',
            'qubit_full_linewidth',
            'amplitude_offset',
        ]
        FittingModelBase.__init__(self, param_names)
    
    # override
    def _model_function(self, drive_frequency, amplitude, qubit_frequency, qubit_full_linewidth, amplitude_offset):
        """Calculate amplitude of cavity response when qubit is driven

        Args:
            drive_frequency (float or np.ndarray): drive frequency of qubit
            amplitude (flaot): peak height, amplitude difference between qubit is in ground and excited state
            qubit_frequency (float): qubit transition frequency
            qubit_full_linewidth (float): half-maximum full linewidth of qubit spectral
            amplitude_offset (float): amplitude offset, response when qubit is in ground state
        
        Returns:
            float or np.ndarray: Amplitud of cavity response
        """
        hwhm = qubit_full_linewidth/2
        detune = drive_frequency - qubit_frequency
        lorentzian = hwhm**2 / (detune**2 + hwhm**2)
        response = amplitude * lorentzian + amplitude_offset
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
        convolve_length = 5
        convolve_window = np.ones(convolve_length)/convolve_length
        convolve_cut = convolve_length//2
        smoothed = np.convolve(y,convolve_window,mode="valid")

        amplitude_offset = np.percentile(smoothed,5)
        max_amplitude = np.max(smoothed)
        amplitude = max_amplitude - amplitude_offset
        max_amplitude_index = np.argmax(smoothed)
        qubit_frequency = x[max_amplitude_index + convolve_cut]

        half_peak_height = amplitude_offset + amplitude/2
        right_half_index = max_amplitude_index + (smoothed<half_peak_height)[max_amplitude_index:].argmax()
        left_half_index = (smoothed>half_peak_height).argmax()
        right_half_freq = x[right_half_index + convolve_cut]
        left_half_freq  = x[left_half_index + convolve_cut]
        qubit_full_linewidth = right_half_freq - left_half_freq

        param_dict = {
            'amplitude'             : amplitude,
            'qubit_frequency'       : qubit_frequency,
            'qubit_full_linewidth'  : qubit_full_linewidth,
            'amplitude_offset'      : amplitude_offset,
        }
        return param_dict
