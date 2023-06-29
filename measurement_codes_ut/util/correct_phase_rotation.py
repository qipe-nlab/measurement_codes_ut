import numpy as np

def correct_phase_rotation(phase, threshold = np.pi):
    """Correct list of phase with mod 2pi to continuous values

    Take difference neighboring phase, and whenever phase changes more than pi,
    add or subtract 2pi to following phases.

    Args:
        phase (np.ndarray): 
        threshold (float): if sample moves more than this value, function assume mod2 moves

    Returns:
        np.ndarray: corrected phase
    """
    diff = np.concatenate([[0],phase[1:] - phase[:-1]])
    warp = np.zeros(phase.shape)
    warp[diff<-threshold]=1
    warp[diff>threshold]=-1
    warp = np.cumsum(warp)
    return phase + warp * np.pi*2
