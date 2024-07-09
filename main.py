import numpy as np
import torch
import NNet
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import time2freq
import F0Table
from FilterBankF0Tracking import f0tracking
import matplotlib.pyplot as plt
from filterh import filterh
'''
rec = pd.read_csv('D:/PHD/Study2/Pilotstudy2_May6_Maryam_righvsLeft/response.csv').values
plt.plot(np.arange(4096)/5714, rec.reshape(-1), label='Left vs Right')
plt.grid()
plt.legend()
stm = pd.read_csv('D:/PHD/Study2/RDstm.csv').values
f0, evalParams = f0tracking(rec.reshape(-1), stm)
'''


F0Table.F0Table('../data2/', 'MS')
# Plots
# h filter heatmap
'''
xl = np.floor((np.arange(10) / 10) * len(fx)).astype(int)
plt.imshow(h, interpolation='nearest', cmap='hot')
plt.xticks(xl, np.floor(fx[xl]).astype(int))
plt.show()


# data heatmap
plt.figure()
plt.imshow(data, interpolation='nearest', cmap='hot')
plt.xticks(xl, np.floor(fx[xl]).astype(int))
plt.show()

plt.figure()
plt.scatter(tx, f0, label='response', color='green')
plt.plot(tx, stm[:, 1], label='Stimulus', color='black')
plt.ylim([0, 500])
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.legend()
'''
# =======================================================================
# =======================================================================
# =======================================================================
'''
device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
subs = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16']
stm = pd.read_csv('./stm/FS.csv').values
# initial parameters
fs = 5714
nfft = 4096
tx = stm[:, 0]
f0min = 80
f0max = 500
k = 4  # number of harmonics
winLen = 50
df = (np.ceil(2 * nfft / winLen) + 1).astype(int)
h, fx = filterh(285, fs, k, 80, 500, nfft=nfft)

# Cut unwanted frequencies
idx_max = np.argmin(np.abs(fx - f0max * k)) + df
idx_min = np.argmax(fx > 60)
fx = fx[idx_min:idx_max]
h = h[:, idx_min:idx_max]
h = torch.from_numpy(h)

data, fx2 = time2freq(record, fs, nfft, 41, 90, 10)
data = data[:stm.shape[0], :]

data = data[:, idx_min:idx_max]
data = torch.from_numpy(data).float()

y = data.unsqueeze(1).repeat(1, h.shape[0], 1).to(torch.float64)
X = data.unsqueeze(1).repeat(1, h.shape[0], 1).to(torch.float64)

X = torch.mul(X, h)
XX = torch.sum(X, dim=2).numpy()
f0_candidates = np.zeros((XX.shape[0], 4))


# Network
model = NNet.Nnet2(h)
model.to(device=device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
'''

# ------------------------------------------------------------
'''
for epoch in range(1000):
    model.train()

    y.to(device=device)
    xp, p = model(data)

    loss = criterion(xp, y)

    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    loss = loss.item()
    if epoch % 50 == 0:
        print(f"loss: {loss:>7f}  [Epoch: {epoch:>5d}]")

'''
