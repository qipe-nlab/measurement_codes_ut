import numpy as np

def moving_average(data,num):
    b    = np.ones(num)/num
    data = np.convolve(data, b, mode='same')
    return data