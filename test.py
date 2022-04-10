import numpy as np
import pandas as pd

import torch

from torch.utils.data import DataLoader

import torchvision
from torchvision.datasets import MNIST

def pil_to_tensor(imgs):
    return torch.from_numpy(np.array(imgs)).unsqueeze(0)
    
def main():
    latent_dim = 2
    mnist_data = MNIST(root='./', download=True)

    loader = DataLoader(mnist_data, 
                        batch_size=8, 
                        shuffle=False, 
                        collate_fn=pil_to_tensor)

    bt = next(iter(loader))

    print(bt.size)

if __name__ == '__main__':
    main()











