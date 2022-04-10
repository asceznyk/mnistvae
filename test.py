import numpy as np
import pandas as pd

import torchvision
from torchvision.datasets import MNIST

mnist_data = MNIST(root='./', download=True)

print(mnist_data[0])









