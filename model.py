import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, img_dim, z_dim):
        super(Encoder, self).__init__()
        self.img_dim = img_dim
        self.z_dim = z_dim

        self.enc = nn.Sequential(
            nn.Conv2d(img_dim[0], 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        self.dense = nn.Linear(64*7*7, 16)
        self.mu = nn.Linear(16, z_dim)
        self.sigma = nn.Linear(16, z_dim)

    def forward(self, x, x_device=device):
        x = self.enc(x)
        x = self.dense(x.view(x.size()[0], -1))
        z_mean = self.mu(x)
        z_std = self.sigma(x)
        eps = torch.normal(0.0, 1.0, size=z_mean.size()).to(x_device)
        z = z_mean + torch.exp(z_std * 0.5) * eps
        return z, z_mean, z_std

class Decoder(nn.Module):
    def __init__(self, img_dim, z_dim):
        super(Decoder, self).__init__()
        self.img_dim = img_dim
        self.z_dim = z_dim

        self.dense = nn.Linear(z_dim, 7*7*64)

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_dim[0], kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        z = self.dense(z)
        x = self.dec(z.view(z.size()[0], 64, 7, 7))
        return x

class VAE(nn.Module):
    def __init__(self, img_dim, z_dim, device):
        super(VAE, self).__init__()
        self.img_dim = img_dim
        self.z_dim = z_dim
        self.device = device

        self.encoder = Encoder(img_dim, z_dim)
        self.decoder = Decoder(img_dim, z_dim)

    def forward(self, x, is_training=False):
        z, z_mean, z_std = self.encoder(x)
        y = self.decoder(z)

        loss_decoder, loss_encoder = None, None 
        if is_training:
            loss_decoder = F.binary_cross_entropy(y, x, reduction='none')
            loss_decoder = torch.mean(torch.sum(loss_decoder, dim=(1,2,3)))
            loss_encoder = -0.5 * (1 + z_std - torch.square(z_mean) - torch.exp(z_std))
            loss_encoder = torch.mean(torch.sum(loss_encoder, dim=-1))
            
        return y, z, loss_decoder, loss_encoder 



