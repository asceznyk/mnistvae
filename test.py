import numpy as np
import pandas as pd

import torch

from torch.utils.data import DataLoader

import torchvision
from torchvision.datasets import MNIST

from model import *

def pil_to_tensor(imgs):
    imgs = np.array([np.array(img) for img, _ in imgs])
    return torch.from_numpy(np.array(imgs)).unsqueeze(1).float()
    
def main():
    latent_dim = 2
    img_dim = (1, 28, 28)
    mnist_data = MNIST(root='./', download=True)

    loader = DataLoader(mnist_data, 
                        batch_size=8, 
                        shuffle=False, 
                        collate_fn=pil_to_tensor)

    encoder = Encoder(img_dim, latent_dim)
    decoder = Decoder(img_dim, latent_dim) 

    z, _, _ = encoder(next(iter(loader)))
    gen = decoder(z)

if __name__ == '__main__':
    main()











