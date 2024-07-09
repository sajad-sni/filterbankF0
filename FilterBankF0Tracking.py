import pandas as pd
import numpy as np
from filterh import filterh
import torch
from scipy import signal
from time2freq import time2freq
import Utilities as F
import matplotlib
import matplotlib.pyplot as plt

# matplotlib.use('Qt5Agg')


def f0tracking(record, stm, sub_idx):
    # initial parameters
    fs = 5714
    nfft = 1024
    tx = stm[:, 0]
    f0min = 80
    f0max = 500
    k = 4  # number of harmonics
    winLen = 285
    df = (np.ceil(2 * nfft / winLen) + 1).astype(int)

    H, fx, scale = filterh(winLen, fs, k, 80, 500, nfft=nfft)
    h = H[-1, :, :]  # harmonic aum
    # Cut unwanted frequencies
    idx_max = np.argmin(np.abs(fx - f0max * k)) + df
    idx_min = np.argmax(fx > 60)
    fx = fx[idx_min:idx_max]
    h = h[:, idx_min:idx_max]
    h = torch.from_numpy(h)

    # Spectrogram
    data, fx2 = time2freq(record, fs, nfft, 41, 90, 10)
    data = data[:stm.shape[0], :]

    data = data[:, idx_min:idx_max]
    data = torch.from_numpy(data).float()

    y = data.unsqueeze(1).repeat(1, h.shape[0], 1).to(torch.float64)
    X = data.unsqueeze(1).repeat(1, h.shape[0], 1).to(torch.float64)

    X = torch.mul(X, h)
    XX = torch.sum(X, dim=2).numpy()
    f0_candidates = np.zeros((XX.shape[0], 4))
    for i in range(XX.shape[0]):
        cand = signal.find_peaks_cwt(XX[i, :], df)
        for j in range(np.min([len(cand), 4])):
            f0_candidates[i, j] = cand[j] + f0min
    f0_idx = np.argmin(np.abs(f0_candidates - stm[:, 1].reshape(-1, 1)), axis=1)
    f0 = f0_candidates[np.arange(XX.shape[0]), f0_idx]
    amp_idx = np.array(f0-f0min).astype(np.int32)
    amp = XX[np.arange(len(amp_idx)), amp_idx]

    f0_amp = F.amp_at_har(X, h, 0, amp_idx)

    # Evaluation
    mse = F.mse(stm[:, 1], f0)
    rmse = F.rmse(stm[:, 1], f0)
    acc5 = F.fivePercent(stm[:, 1], f0)

    fig, ax = plt.subplots(3, 1)
    ax[0].scatter(tx, f0, label='response', color='green', s=3)
    ax[0].plot(tx, stm[:, 1], label='Stimulus', color='black')
    ax[0].set_ylim([0, 500])
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Frequency')
    ax[0].set_title('Subject {}'.format(sub_idx+1))

    ax[0].legend()

    ax[1].plot(tx, amp, label='response', color='green')
    # ax[1].set_ylim([0, 1])
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Amplitude')

    ax[2].imshow(data, cmap='hot')
    plt.show()
    return f0, amp, (mse, rmse, acc5), fig, f0_amp
