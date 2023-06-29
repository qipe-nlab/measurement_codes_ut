import numpy as np

def _process_edge(signal, window_size, edge_process):
    """Processing edge of convoluted signal

    Args:
        signal (np.ndarray): signal to convolute
        width_size (int): convolution width
        edge_process (str): how to process convolution edge. "same": filling with nearest value, "zero": filling with zeros, "empty": return shorter array

    Retruns:
        np.ndarray: processed signal
    """
    processed_signal = None
    if edge_process == "same":
        processed_signal = np.concatenate([
            np.tile(signal[0],window_size),
            signal,
            np.tile(signal[-1],window_size)
        ])

    elif edge_process == "zero":
        processed_signal = np.concatenate([
            np.zeros(window_size, dtype = signal.dtype),
            signal,
            np.zeros(window_size, dtype = signal.dtype)
        ])
    
    elif edge_process == "empty":
        processed_signal = signal

    else:
        raise ValueError("Unknown edge processing type {}. Choose from same, zero, or empty.".format(edge_process))

    return processed_signal

def convolve(signal, window, edge_process = "same"):
    """Convolute signal with a given window

    Args:
        signal (np.ndarray): signal to convolute
        window_size (int): width to take convolution. default to 1.
        edge_process (str): how to process convolution edge. "same": filling with nearest value, "zero": filling with zeros, "empty": return shorter array

    Returns:
        np.ndarray: processed signal
    """
    processed_signal = np.convolve(signal,window,mode="valid")
    processed_signal_with_edge = _process_edge(processed_signal, len(window), edge_process)
    return processed_signal_with_edge


def convolve_flat(signal, window_size = 1, edge_process = "same"):
    """Convolute signals with flat window

    Args:
        signal (np.ndarray): signal to convolute
        window_size (int): width to take convolution. default to 1.
        edge_process (str): how to process convolution edge. "same": filling with nearest value, "zero": filling with zeros, "empty": return shorter array

    Returns:
        np.ndarray: processed signal
    """
    window = np.ones(window_size)/window_size
    return convolve(signal, window, edge_process)


def convolve_gauss(signal, window_size = 5, gaussian_width = 1, edge_process = "same"):
    """Convolute signals with Gaussian window

    Args:
        signal (np.ndarray): signal to convolute
        window_half_width (int): width to take convolution. default to 1.
        gaussian_width (int): width of gaussian window

    Returns:
        np.ndarray: processed signal
    """
    x = np.arange(window_size,dtype=float)-(window_size-1)/2
    unnormalized_window = np.exp(-(x/gaussian_width)**2)
    window = unnormalized_window / np.sum(unnormalized_window)
    return convolve(signal, window, edge_process)
