#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 19:16:55 2023

@author: 
"""

from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from utils import img_size

class CommonDataset(Dataset):
    def __init__(self, data_dir, name = None, mode = False):
        """
        mode True --> training, 
        mode False --> test
        
        """
        
        self.path = data_dir
        self.name = name 
        if name == 'mnist':
            self.dataset = torchvision.datasets.MNIST(self.path, train = mode, download = True)
            self.classes = self.dataset.classes
        elif name == 'cifar10':
            self.dataset = torchvision.datasets.CIFAR10(self.path, train = mode, download = True)
            self.classes = self.dataset.classes
        elif name == 'fashionmnist': 
            self.dataset = torchvision.datasets.FashionMNIST(self.path, train = mode, download = True)
            self.classes = self.dataset.classes
        elif name == 'country211': 
            mode = "train" if mode == True else "test"
            self.dataset = torchvision.datasets.Country211(self.path, split = mode, download = True)
            self.classes = self.dataset.classes
        self.transform_operation_cifar = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize([int(np.sqrt(img_size)),int(np.sqrt(img_size))]),
            transforms.ToTensor(),
            ])
        self.transform_operation = transforms.Compose([
            #transforms.ToPILImage(),
            #transforms.Grayscale(num_output_channels=1),
            transforms.Resize([int(np.sqrt(img_size)),int(np.sqrt(img_size))]),
            #transforms.ToTensor(),
            ])
        
    def __len__(self):
        return len(self.dataset)  
    
    def __getitem__(self,idx):
        
        if self.name == "country211":
            imgs, clss =  self.transform_operation_211(self.dataset[idx][0]), self.dataset[idx][1]
        elif self.name == "cifar10":
            imgs, clss =  self.transform_operation_cifar(self.dataset.data[idx]), self.dataset[idx][1]
        else:
            imgs, clss = self.transform_operation(self.dataset.data[idx].unsqueeze(0)), self.dataset.targets[idx]
        
        return imgs, clss
    

def call_dataloader(path, batch_size, name):
    
    #create dataset object
    train_dataset = CommonDataset(path, name = name, mode = True)
    test_dataset = CommonDataset(path, name = name, mode = False)
    
    trainloader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

    return trainloader, test_loader

if __name__ == "__main__":
    trainloader, test_loader = call_dataloader("./data", batch_size = 32, name = "cifar10")