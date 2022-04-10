import numpy as np
import pandas as pd

import torchvision
from torchvision.datasets import MNIST

def pil_to_tensor(img):
    print(torch.from_numpy(img))
    return

def main():
    latent_dim = 2
    mnist_data = MNIST(root='./', download=True)

if __name__ == '__main__':
    main()











