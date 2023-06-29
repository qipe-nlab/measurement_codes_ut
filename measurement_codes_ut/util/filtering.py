
import numpy as np
import scipy.signal

def lowpass_filter(signal, period, order = 4):
    """Lowpass filtering

    Removing oscillation with period shorter than a given value

    Args:
        signal (np.ndarray): signal to process
        period (float): sample number for each period
        order (int): order of Butterworth filter
    
    Returns:
        np.ndarray: processed signal
    """
    b,a = scipy.signal.butter(order, 2/period, "lowpass")
    processed_signal = scipy.signal.filtfilt(b,a,signal)
    return processed_signal

def highpass_filter(signal, period, order = 4):
    """Highpass filtering

    Removing oscillation with period longer than a given value

    Args:
        signal (np.ndarray): signal to process
        period (float): sample number for each period
        order (int): order of Butterworth filter
    
    Returns:
        np.ndarray: processed signal
    """
    b,a = scipy.signal.butter(order, 2/period, "highpass")
    processed_signal = scipy.signal.filtfilt(b,a,signal)
    return processed_signal

def bandpass_filter(signal, min_period, max_period, order = 2):
    """Bandpass filtering

    Removing oscillation with period longer than max_period or shorter than min_period

    Args:
        signal (np.ndarray): signal to process
        min_period (float): sample number for minimum period
        max_period (float): sample number for maximumperiod
        order (int): order of Butterworth filter
    
    Returns:
        np.ndarray: processed signal
    """
    b,a = scipy.signal.butter(order, [2/max_period, 2/min_period], "bandpass")
    processed_signal = scipy.signal.filtfilt(b,a,signal)
    return processed_signal

def bandstop_filter(signal, min_period, max_period, order = 2):
    """Bandstop filtering

    Removing oscillation with period longer than min_period and shorter than max_period

    Args:
        signal (np.ndarray): signal to process
        min_period (float): sample number for minimum period
        max_period (float): sample number for maximumperiod
        order (int): order of Butterworth filter
    
    Returns:
        np.ndarray: processed signal
    """
    b,a = scipy.signal.butter(order, [2/max_period, 2/min_period], "bandstop")
    processed_signal = scipy.signal.filtfilt(b,a,signal)
    return processed_signal

