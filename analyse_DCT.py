import numpy as np
import scipy
import scipy.fft as fft
from scipy import signal


def dct2(frames):
    """
    Computes the DCT2 transform for all frames.
    """
    N = np.shape(frames)[-2]
    inv_norm = 1 / (2 * N)
    return inv_norm * fft.dct(frames, axis=-2)

def idct(frames):
    """
    Computes the inverse DCT2 transform for all frames.
    """
    N = np.shape(frames)[-2]
    norm = 2 * N
    return norm * fft.idct(frames, axis=-2)
    

def modes2(frames):
    """
    Computes the mean square average of the Rouse modes for all frames

    frames.shape = (Ns, N, 3)
    where
        Ns: frames number   (axis=-3)
        N: signal size     (axis=-2)
        3: space dimension (axis=-1)
    """

    return np.square(np.abs(dct2(frames))).mean(axis=-3).sum(axis=-1)




import json
def extract(name):
    frames = []
    with open(name) as file:
        frames.extend(map(json.loads, file))

    
    frames = np.array([np.array(frame) for frame in frames])
    return frames

