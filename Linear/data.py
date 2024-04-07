"""   Contains the code for loading the dataloaders of CIFAR-10 and CIFAR-100     """


import os
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10,CIFAR100

import random
from torchvision import transforms
import numpy as np


class Dataset(pl.LightningDataModule):
    def __init__(self, data_dir, dataset):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = dataset

    ### CIFAR-10 and CIFAR-100 dataloading adapted from: https://github.com/val-iisc/NuAT/blob/main/CIFAR10/NuAT_cifar10.py
    def cifar10_dataloaders(self):
        transform_train = transforms.Compose([
                transforms.RandomCrop(size=32,padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),])
        transform_test = transforms.Compose([
                transforms.ToTensor(),])
                
        train_set  = CIFAR10(root='./data', train=True , download=True, transform=transform_train)
        test_set   = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        
        train_loader = DataLoader(train_set,batch_size=512, shuffle=True)
        test_loader   = DataLoader(test_set,batch_size=512)
        print('CIFAR10 dataloader: Done') 
        return train_loader, test_loader


    def cifar100_dataloaders(self):
        transform_train = transforms.Compose([
                transforms.RandomCrop(size=32,padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),])
        transform_test = transforms.Compose([
                transforms.ToTensor(),])
                
        train_set  = CIFAR100(root='./data', train=True , download=True, transform=transform_train)
        test_set   = CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        
        train_loader = DataLoader(train_set,batch_size=512, shuffle=True)
        test_loader  = DataLoader(test_set, batch_size=512)
        print('CIFAR100 dataloader: Done') 
        return train_loader, test_loader
