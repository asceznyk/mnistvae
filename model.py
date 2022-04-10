import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, img_dim, z_dim):
        super(Encoder,self).__init__()
        self.img_dim = img_dim
        self.z_dim = z_dim

        self.enc = nn.Sequential(
            nn.Conv2d(img_dim[0], 32, 3, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2),
            nn.ReLU(),
            nn.Linear(64*7*7, 16)
        )

        self.mu = nn.Linear(16, z_dim)
        self.sigma = nn.Linear(16, z_dim)

    def forward(self, x):
        x = self.enc(x)
        z_mean = self.mu(x)
        z_std = self.sigma(x)
        z = torch.normal(z_mean, z_std)
        return z, z_mean, z_std










