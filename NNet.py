import torch
from torch import nn

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

'''
class Nnet(nn.Module):
    def __init__(self, H, X, f0min, f0max, fx, k):
        super().__init__()
        self.f0min = f0min
        self.f0max = f0max
        self.k = k
        self.fx = fx
        self.h = h
        self.fres = fx[2]-fx[1]
        self.fm = fx[-1]
        for i, f0 in enumerate(range(f0min, f0max)):
            layer = 'conv' + str(i)
            setattr(self, layer, nn.Conv1d(1, 1, len(h), padding=0, stride=f0))
            weights = torch.zeros_like(getattr(self, layer).weight)
            weights[-1] = h
            getattr(self, layer).weight = nn.Parameter(weights)
            self.relu = nn.ReLU()

    def forward(self, x):
        for i, f0 in enumerate(range(self.f0min, self.f0max)):
            start_idx = (torch.abs(self.fx - f0)).argmin()
            stop_idx = (torch.abs(self.fx - self.k*f0)).argmin()
            # start_idx = f0 / self.fres - round(len(self.h) / 2)
            # stop_idx = self.k*f0/self.fres + round(len(self.h)/2)
            conv = getattr(self, 'conv' + str(i))
            x = conv(x[start_idx:stop_idx])
            x = self.relu(x)
            y = torch.zeros_like(self.x)
'''


class Nnet2(nn.Module):
    def __init__(self, h):
        super().__init__()
        # h shape: f0 candidates * fftlen
        self.h = h
        self.mat = nn.Parameter(torch.rand_like(h))
        self.float()

    def forward(self, x):
        x = torch.mul(x, self.mat)
        p = torch.sum(x, dim=2)
        return x, p







