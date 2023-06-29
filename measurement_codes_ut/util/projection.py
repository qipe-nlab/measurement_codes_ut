
from sklearn.decomposition import PCA

def project_iq_signal(iq_signal):
    """Project a given signal to main and anti axis

    Args:
        iq_signal (np.ndarray): 2-dimensional array
    Returns:
        np.ndarray: result of PCA
    """
    model = PCA()
    proj_signal = model.fit_transform(iq_signal)
    return proj_signal

