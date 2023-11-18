#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 22:36:43 2023

@author: 
"""

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
import matplotlib.pyplot as plt
import pickle
import random
import numpy as np
from Dataset import call_dataloader



from models import get_trained_model
from ts_walsh import get_values, get_tse
learning_rate = 3e-5
device_ = "cpu" # "cuda:0" "cuda:1" "cpu"

def save_the_model(name, network):
    
    
    path = "./outputs/saved_model/"+name+".pickle"
    
    with open(path, 'wb') as f:
        pickle.dump(network.dnn, f)

    

def set_seed(seed = 42):
    
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_iterations(name, ltt):
    
    path = "./outputs/training_results/"+name+".pickle"
    
    with open(path, 'wb') as f:
        pickle.dump(record, f)

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description='Pytorch Pretrained Model Comparison')
    parser.add_argument('--model_name', type=str, default='base',
                        help='resnet50, resnet34, vgg19, vgg16, googleNet, mobilnet, squeezenet, inception (default: alexnet)')
    parser.add_argument('--epochs', type=int, default=2,
                        help='32, 64, 128 (default: 64)')   
    parser.add_argument('--dataset', type=str, default = "cifar10", 
                        help='mnist, cifar10, fashionmnist, country211')
    args = parser.parse_args()

    #
    set_seed()
    
    #

    if device_ != None:
        device = torch.device(device_)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #

    trainloader, test_loader = call_dataloader("./data", batch_size = 200, name = args.dataset)

    network = get_trained_model("combined_"+args.model_name)
    network.to(device)
    criterion = torch.nn.CrossEntropyLoss()  #KLDivLoss , CrossEntropyLoss
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    network.train()
    targets = get_values()
    one_hot_vectors = torch.nn.functional.one_hot(targets, 10).float()
    loss_max = 9*10**7
    loss_values = []
    epochs = int(args.epochs)
    for i in range(epochs):
        print("epochs ", i)
        for loader in trainloader:
            print(args.dataset)
            x_tsne, targets =  get_tse((loader))
            x_tsne = x_tsne.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            output = network(x_tsne, targets)
            loss = criterion(output, targets)
            loss_values.append(loss.item())

            if loss_max > loss:
                loss_max = loss
                save_the_model(args.model_name+args.dataset, network)
            loss.backward()
            optimizer.step()
    network.eval()

    # print(all(network.dnn.fc3.weight == initial_weights))

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=500)

    plt.title("T-NSE Empowered " + args.model_name +  " Network Pre-training in " + args.dataset)

    color = 'tab:purple'
    plt.plot(loss_values, color=color)
    fig.savefig("./outputs/history_values/" + args.model_name+args.dataset + ".jpg" , dpi=1000)
       #Save values

    name = "combined_"+args.model_name+args.dataset + "Pre-training_model_prf_results"
    record = {
        'loss_values':loss_values,
      }
    
    save_iterations(name, record)
    # save_the_model("baseline", network)
