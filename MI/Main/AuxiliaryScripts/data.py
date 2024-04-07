import os
import tarfile
import random
import requests
from typing import Union

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10,CIFAR100
from torchvision import transforms
from torchvision.datasets import ImageFolder

import numpy as np
import pytorch_lightning as pl
from AuxiliaryScripts import DataGenerator as DG

class Dataset(pl.LightningDataModule):
    def __init__(self, data_dir, dataset, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = dataset
        self.batch_size = batch_size
        # self.mean = (0.4914, 0.4822, 0.4465)
        # self.std = (0.2471, 0.2435, 0.2616)
        


    ### CIFAR-10 and CIFAR-100 dataloading adapted from: https://github.com/val-iisc/NuAT/blob/main/CIFAR10/NuAT_cifar10.py
    def cifar10_dataloaders(self):
        transform_train = transforms.Compose([
                transforms.RandomCrop(size=32,padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),])
        transform_test = transforms.Compose([
                transforms.ToTensor(),])
                
        train_set  = CIFAR10(root='../data', train=True , download=True, transform=transform_train)
        test_set   = CIFAR10(root='../data', train=False, download=True, transform=transform_test)
        
        train_loader = DataLoader(train_set,batch_size=self.batch_size, shuffle=True)
        test_loader   = DataLoader(test_set,batch_size=self.batch_size)
        print('CIFAR10 dataloader: Done') 
        return train_loader, test_loader


    def cifar100_dataloaders(self):
        transform_train = transforms.Compose([
                transforms.RandomCrop(size=32,padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),])
        transform_test = transforms.Compose([
                transforms.ToTensor(),])
                
        train_set  = CIFAR100(root='../data', train=True , download=False, transform=transform_train)
        test_set   = CIFAR100(root='../data', train=False, download=False, transform=transform_test)
        
        train_loader = DataLoader(train_set,batch_size=self.batch_size, shuffle=True)
        test_loader  = DataLoader(test_set, batch_size=self.batch_size)
        print('CIFAR100 dataloader: Done') 
        return train_loader, test_loader


    def tinyimagenet_dataloaders(self):
        ### Data is already processed without normalizing mean and std, only converting to tensor in range [0,1]
        
        x_train =torch.load('../data/Tiny Imagenet/train/X.pt')
        y_train = torch.load('../data/Tiny Imagenet/train/y.pt')
        x_test = torch.load('../data/Tiny Imagenet/val/X.pt')
        y_test = torch.load('../data/Tiny Imagenet/val/y.pt')
        
        ### Makes a custom dataset for a given dataset through torch
        train_set = DG.DataGenerator(x_train,y_train)
        test_set = DG.DataGenerator(x_test,y_test)
    
        ### Loads the custom data into the dataloader
        train_loader =  DataLoader(train_set, batch_size = self.batch_size, shuffle = True)
        test_loader =  DataLoader(test_set, batch_size = self.batch_size, shuffle = False)
        
        return train_loader, test_loader












    def cifar10_dataset(self):
        transform_train = transforms.Compose([
                transforms.RandomCrop(size=32,padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),])
        transform_test = transforms.Compose([
                transforms.ToTensor(),])
                
        train_set  = CIFAR10(root='../data', train=True , download=False, transform=transform_train)
        test_set   = CIFAR10(root='../data', train=False, download=False, transform=transform_test)
        
        return train_set, test_set


    def cifar100_dataset(self):
        transform_train = transforms.Compose([
                transforms.RandomCrop(size=32,padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),])
        transform_test = transforms.Compose([
                transforms.ToTensor(),])
                
        train_set  = CIFAR100(root='../data', train=True , download=True, transform=transform_train)
        test_set   = CIFAR100(root='../data', train=False, download=True, transform=transform_test)
        
        return train_set, test_set
        
        
        
        
        



