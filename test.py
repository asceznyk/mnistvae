import numpy as np
import pandas as pd

import torch

from torch.utils.data import DataLoader

import torchvision
from torchvision.datasets import MNIST
from torchvision.utils import make_grid 

from utils import *
from model import *

def pil_to_tensor(imgs):
    imgs = np.array([np.array(img)/255.0 for img, _ in imgs])
    return torch.from_numpy(np.array(imgs)).unsqueeze(1).float()
    
def main():
    batch_size = 128
    latent_dim = 2
    img_dim = (1, 28, 28)
    mnist_data = MNIST(root='./', download=True)

    loader = DataLoader(mnist_data, 
                        batch_size=batch_size, 
                        shuffle=True, 
                        collate_fn=pil_to_tensor)

    to_show = next(iter(loader))
    vae = VAE(img_dim, latent_dim)
    y, z, ld, le = vae(to_show, is_training=True)

    print(ld, le)

    show(make_grid(to_show, nrows=batch_size))
    show(make_grid(y, nrows=batch_size), 'y_num.png')

    print(vae)
    print(y)
    print(z)


if __name__ == '__main__':
    main()











