import scipy
import numpy as np
import Utilities as F
import matplotlib.pyplot as plt


def filterh(winlen, fs, k, f0min, f0max, nfft):
    win = scipy.signal.windows.hamming(winlen)
    h = np.zeros((k+1, f0max - f0min + 1, nfft // 2+1))
    h = np.zeros((k+1, f0max - f0min + 1, nfft // 2))
    t = np.arange(0, winlen/1000, 1/fs)
    t = np.arange(0, winlen)/fs
    norm = 0
    for i, f in enumerate(range(f0min, f0max+1)):
        s = 0
        for har_idx, har in enumerate(range(1, k+1)):
            #harmonic = np.sin(2*np.pi*har*f*t)
            harmonic = np.exp(1j*2*np.pi*har*f*t)
            har_fft = np.abs(np.fft.fft(harmonic, nfft)[:nfft//2])
            har_fft[1:] = har_fft[1:]*2
            h[har_idx, f-f0min, :] = har_fft/k
            s += harmonic

        # s0, fx0 = F.fft_onesided(s, fs, nfft)
        s = s/k
        sf = np.fft.fft(s, nfft)[:nfft//2]
        fx = np.fft.fftfreq(nfft, 1/fs)[:nfft//2]
        sf = np.abs(sf)
        sf[1:] = sf[1:]*2
        norm = (i * norm + np.max(sf)) / (i + 1)
        h[-1, f-f0min, :] = sf

    h[h < 90] = 0
    # Calculate the sum of each row in the b*c matrix
    h_sum = np.sum(h**2, axis=2, keepdims=True)
    h_sum = np.mean(h_sum, axis=1, keepdims=True)
    # Divide each element in the row by the corresponding row sum
    h = h / h_sum

    return h, fx, 1/norm

