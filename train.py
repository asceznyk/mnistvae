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

def pil_to_tensor(batch):
    imgs = np.array([np.array(img)/255.0 for img, _ in batch])
    labels = torch.from_numpy(np.array([b[1] for b in batch])).float()
    return torch.from_numpy(np.array(imgs)).unsqueeze(1).float(), labels

def main():
    epochs = 30
    batch_size = 128
    latent_dim = 2
    img_dim = (1, 28, 28)
    is_train = True
    ckpt_path='vae_mnist.ckpt'

    mnist_data = MNIST(root='./', download=True)

    loader = DataLoader(mnist_data, 
                        batch_size=batch_size, 
                        shuffle=True, 
                        collate_fn=pil_to_tensor)

    vae = VAE(img_dim, latent_dim, device)

    vae.train(is_train)
    best_loss = float('inf') 
    optimizer = torch.optim.Adam(vae.parameters()) 

    for e in range(1, epochs+1):
        vae.to(device)

        avg_loss = 0
        pbar = tqdm(enumerate(loader), total=len(loader))
        for i, (imgs, labels) in pbar: 
            imgs = imgs.to(device)
            with torch.set_grad_enabled(is_train):  
                y, z, loss_recons, loss_kl = vae(imgs, is_training=is_train)
                loss = loss_recons + loss_kl 
                avg_loss += loss.item() / len(loader)

            if is_train:
                vae.zero_grad() 
                loss.backward() 
                optimizer.step()

            pbar.set_description(f"epoch: {e}, loss: {loss.item():.3f}, avg: {avg_loss:.2f}") 

        view_predict(vae)
        view_label_clusters(vae, loader)

        if ckpt_path is not None and avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(vae.state_dict(), ckpt_path)

if __name__ == '__main__':
    main()


