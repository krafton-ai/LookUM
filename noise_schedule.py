import abc
import torch
import torch.nn as nn

class Noise(abc.ABC, nn.Module):

    def forward(self, t):
        return (self.total_noise(t), self.rate_noise(t))

    @abc.abstractmethod
    def rate_noise(self, t):
        pass

    @abc.abstractmethod
    def total_noise(self, t):
        pass

class LogLinearNoise(Noise):

    def __init__(self, eps=0.001):
        super().__init__()
        self.eps = eps
        self.sigma_max = self.total_noise(torch.tensor(1.0))
        self.sigma_min = self.eps + self.total_noise(torch.tensor(0.0))

    def rate_noise(self, t):
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total_noise(self, t):
        return -torch.log1p(-(1 - self.eps) * t)

def get_noise(config, dtype=torch.float32):
    if config.noise.type != 'loglinear':
        raise ValueError('Expected loglinear noise.')
    return LogLinearNoise()
