import numpy as np
import pandas as pd
import torch

def fivePercent(ref, sig):
    valid = np.abs(ref - sig) < 0.05 * ref
    return 100 * sum(valid) / len(ref)


def mse(ref, sig):
    return (np.square(ref - sig)).mean()

def mse_relative(ref, sig):
    return (np.square(np.divide((ref - sig), ref))).mean()


def rmse(ref, sig):
    return np.sqrt((np.square(ref - sig)).mean())


def rmse_trlative(ref, sig):
    return np.sqrt((np.square(np.divide((ref - sig), ref))).mean())


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


def amp_mse(stm_path, rec_path):
    mse_list = []
    rmse_list = []
    stm = pd.read_excel(stm_path, index_col=0).values
    rec = pd.read_excel(rec_path, index_col=0).values
    for i in range(rec.shape[0]):
        mse_list += [mse_relative(stm, rec[i])]
        rmse_list += [rmse_trlative(stm, rec[i])]

    subs = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15',
            'S16']

    result = pd.DataFrame({
        "MSE amp @ f0": mse_list,
        "RMSE amp @ f0": rmse_list
    }, index=subs)

    result.to_excel('amp_f0_analysis.xlsx', index=True, header=True)


def amp_ratio(stm_path, rec_path):
    stm = pd.read_excel(stm_path, index_col=0).values
    rec = pd.read_excel(rec_path, index_col=0).values

    ratio = rec/stm

    print(f"mean: {np.mean(ratio)}\n std: {np.mean(np.std(ratio, axis=0))}")
