import numpy as np
import pandas as pd

from tqdm import tqdm

import torch

from torch.utils.data import DataLoader

import torchvision
from torchvision.datasets import MNIST
from torchvision.utils import make_grid 

from utils import *
from model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pil_to_tensor(imgs):
    imgs = np.array([np.array(img)/255.0 for img, _ in imgs])
    return torch.from_numpy(np.array(imgs)).unsqueeze(1).float()

def fit(model, train_loader, valid_loader=None, ckpt_path=None, epochs=10):  
    def run_epoch(split):
        is_train = split == 'train' 
        model.train(is_train)
        loader = train_loader if is_train else valid_loader

        avg_loss = 0
        pbar = tqdm(enumerate(loader), total=len(loader))
        for imgs in pbar: 
            print(imgs)
            imgs = imgs.to(device)
            with torch.set_grad_enabled(is_train):  
                y, z, loss_enc, loss_dec = model(imgs)
                loss = loss_enc + loss_dec
                avg_loss += loss.item() / len(loader)

            if is_train:
                model.zero_grad() 
                loss.backward() 
                optimizer.step()

            pbar.set_description(f"epoch: {e}, loss: {loss.item():.3f}, avg: {avg_loss:.2f}")     
        return avg_loss

    model.to(device)

    best_loss = float('inf') 
    optimizer = torch.optim.Adam(model.parameters()) 
    for e in range(1, epochs+1):
        train_loss = run_epoch('train')
        valid_loss = run_epoch('valid') if valid_loader is not None else train_loss

        if ckpt_path is not None and valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), ckpt_path)

def main():
    epochs = 30
    batch_size = 128
    latent_dim = 2
    img_dim = (1, 28, 28)
    is_training = True

    mnist_data = MNIST(root='./', download=True)

    loader = DataLoader(mnist_data, 
                        batch_size=batch_size, 
                        shuffle=True, 
                        collate_fn=pil_to_tensor)
    
    vae = VAE(img_dim, latent_dim)
    fit(vae, loader, ckpt_path='vae_mnist.ckpt', epochs=epochs)

if __name__ == '__main__':
    main()


