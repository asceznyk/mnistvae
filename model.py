import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, img_dim, z_dim):
        super(Encoder, self).__init__()
        self.img_dim = img_dim
        self.z_dim = z_dim

        self.enc = nn.Sequential(
            nn.Conv2d(img_dim[0], 32, kernel_size=3, stride=2, pad=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, pad=1),
            nn.ReLU(),
        )

        self.dense = nn.Linear(64*7*7, 16)
        self.mu = nn.Linear(16, z_dim)
        self.sigma = nn.Linear(16, z_dim)

    def forward(self, x):
        x = self.enc(x)
        print(x.size())
        x = self.dense(x.view(x.size()[0], -1))
        z_mean = self.mu(x)
        z_std = self.sigma(x)
        eps = torch.normal(0.0, 1.0, size=z_mean.size())
        z = z_mean + torch.exp(z_std * 0.5) * eps
        return z, z_mean, z_std

class Decoder(nn.Module):
    def __init__(self, img_dim, z_dim):
        super(Decoder, self).__init__()
        self.img_dim = img_dim
        self.z_dim = z_dim

        self.dense = nn.Linear(z_dim, 6*6*64)

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d()
        )





