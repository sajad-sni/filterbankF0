import scipy
import torch
from scipy import signal
import numpy as np
import Utilities as F
import math

# cut: cutting time(ms)
# winlen = frame time(ms)
# step = frame overlap(ms)


def time2freq(s, fs, nfft, cut, winlen, step):
    fsample = round(winlen * fs/1000)
    step = round(step*fs/1000)

    s = s[round(cut*fs/1000):]
    s = torch.from_numpy(s)
    s = s.unfold(0, fsample, step).numpy()
    row, col = np.shape(s)
    w = signal.windows.hamming(col)
    s = np.multiply(s, w)
    # z = np.zeros((row, nfft - col))
    # s = np.concatenate((s, z), axis=1)
    L = s.shape[1]
    s = np.fft.fft(s, nfft, axis=1)[:, :nfft // 2] / L

    fx = np.fft.fftfreq(nfft, d=1/fs)[:nfft//2]
    # fx2 = np.arange(nfft)*fs/nfft
    # fx3 = np.arange(0,fs,  fx[1])
    #s = s[:, 0: len(fx)]
    s = np.abs(s)
    s[:, 1:] = s[:, 1:] * 2
    return s, fx



