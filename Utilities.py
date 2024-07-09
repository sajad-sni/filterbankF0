import numpy as np
import pandas as pd
import torch

def fivePercent(ref, sig):
    valid = np.abs(ref - sig) < 0.05 * ref
    return 100 * sum(valid) / len(ref)


def mse(ref, sig):
    return (np.square(ref - sig)).mean()


def rmse(ref, sig):
    return np.sqrt((np.square(ref - sig)).mean())


def fft_onesided(x, fs, n):
    # Compute the FFT
    Y = np.fft.fft(x, n)
    P2 = np.abs(Y / len(x))  # Two-sided spectrum

    # Compute the one-sided spectrum
    P1 = P2[:n // 2 + 1]
    P1[1:-1] = 2 * P1[1:-1]  # Double the amplitudes (except the DC and Nyquist)

    # Define the frequency domain f
    f = np.fft.fftfreq(n, 1 / fs)[:n // 2 + 1]
    return P1, f


def amp_at_har(X, h, har, idx):
    X = torch.mul(X, h[har])
    XX = torch.sum(X, dim=2).numpy()
    har_amp = XX[np.arange(len(idx)), idx]
    return har_amp
