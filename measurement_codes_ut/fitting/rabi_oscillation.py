

import numpy as np
from scipy import signal
from .base import FittingModelBase


class RabiOscillation(FittingModelBase):
    """Fitting model for rabi oscillation
    """
    # override
    def __init__(self):
        """Initializer of rabi oscillation
        """
        param_names = [
            'amplitude',
            'rabi_frequency',
            'phase_offset',
            'decay_rate',
            'amplitude_offset',
        ]
        FittingModelBase.__init__(self, param_names)
    
    # override
    def _model_function(self, time, amplitude, rabi_frequency, phase_offset, decay_rate, amplitude_offset):
        """Response function for rabi oscillation

        Args:
            time (float or np.ndarray): driving time of qubit
            amplitude (float): amplitude of rabi oscillation
            rabi_frequency (float): rabi frequency
            phase_offset (float): phase offset of rabi oscillation
            decay_rate (float): decaying rate of rabi oscillation
            amplitude_offset (float): amplitude offset of rabi oscillation
        
        Returns:
            float or np.ndarray: rabi oscillation signal
        """
        oscillation = np.cos(phase_offset + 2 * np.pi * rabi_frequency * time)
        decay_term = np.exp( - time * decay_rate)
        response = amplitude_offset + amplitude * oscillation * decay_term
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
        # amplitude of sine-wave is (max-min)/2
        amplitude = (np.max(y) - np.min(y))/2

        # take two abs_peak  and their points, and compute decay rate.
        # exp(-decay_rate * time)
        half_data_count = len(x)//2
        peak = np.max(np.abs(y))
        peak_index = np.argmax(np.abs(y))
        peak_after_half = np.max(np.abs(y[half_data_count:]))
        peak_after_half_index = half_data_count + np.argmax(np.abs(y[half_data_count:]))
        half_time = x[peak_after_half_index] - x[peak_index]

        if peak>peak_after_half:
            decay_rate = np.log(peak/peak_after_half) / half_time
        else:
            decay_rate = 0.1

        # use LPF and take local minimums
        b,a = signal.butter(N=10,Wn=0.3,output='ba')
        y_lpf = signal.filtfilt(b,a,y)
        minimum_index_list = signal.argrelmin(y_lpf, order=5)[0]
        first_minimum_time = x[minimum_index_list[0]]
        second_minimum_time = x[minimum_index_list[1]]
        rabi_frequency = 1. / (second_minimum_time - first_minimum_time)
        param_dict = {
            'amplitude'         : amplitude,
            'rabi_frequency'    : rabi_frequency,
            'phase_offset'      : 0.,
            'decay_rate'        : decay_rate,
            'amplitude_offset'  : 0.,
        }
        return param_dict
