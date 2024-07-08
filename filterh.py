import scipy
import numpy as np
import Utilities as F
import matplotlib.pyplot as plt


def filterh(winlen, fs, k, f0min, f0max, nfft):
    win = scipy.signal.windows.hamming(winlen)
    h = np.zeros((f0max - f0min + 1, nfft // 2+1))
    h = np.zeros((f0max - f0min + 1, nfft // 2))
    t = np.arange(0, winlen/1000, 1/fs)
    t = np.arange(0, winlen)/fs
    norm = 0
    for i, f in enumerate(range(f0min, f0max)):
        s = 0
        for har in range(1, k+1):
            #s += np.sin(2*np.pi*har*f*t)
            s += np.exp(1j*2*np.pi*har*f*t)
        # s0, fx0 = F.fft_onesided(s, fs, nfft)
        s = s/k
        sf = np.fft.fft(s, nfft)[:nfft//2]
        fx = np.fft.fftfreq(nfft, 1/fs)[:nfft//2]
        sf = np.abs(sf)
        sf[1:] = sf[1:]*2
        norm = (i * norm + np.max(sf)) / (i + 1)
        h[f-f0min, :] = sf
    h[h < 90] = 0
    return h, fx, 1/norm

