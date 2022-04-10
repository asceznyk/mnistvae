import numpy as np
import pandas as pd

import torch

import torchvision
from torchvision.datasets import MNIST

def pil_to_tensor(img):
    print(torch.from_numpy(np.array(img)).unsqueeze(-1).size())
    return torch.from_numpy(np.array(img))
    
def main():
    latent_dim = 2
    mnist_data = MNIST(root='./', download=True)

    img, _ = mnist_data[0]

    pil_to_tensor(img)

if __name__ == '__main__':
    main()











