#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 22:36:43 2023

@author: talha
"""

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
import matplotlib.pyplot as plt
import pickle

from models import get_trained_model
from ts_walsh import get_values
learning_rate = 0.001
epochs = 1000


def save_the_model(name, network):
    
    path = "./outputs/saved_model/"+name+".pickle"
    
    with open(path, 'wb') as f:
        pickle.dump(network.dnn, f)

def load_the_model(name):
    
    path = "./outputs/saved_model/"+name+".pickle"
    
    with open(path, 'rb') as file:
        model = pickle.load(file)
    
    return model
    

name = "combined_base"
network = get_trained_model(name)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(network.parameters(), lr=learning_rate)

initial_weights = network.dnn.fc3.weight.clone()
network.train()
x_tsne, targets = get_values()
loss_values = []
for i in range(epochs):
        optimizer.zero_grad()
        output = network(x_tsne, targets)
        loss = criterion(output, targets)
        loss_values.append(loss.item())
        loss.backward()
        optimizer.step()
network.eval()

# print(all(network.dnn.fc3.weight == initial_weights))

fig, ax1 = plt.subplots(figsize=(10, 6), dpi=500)

plt.title("Support "+ name + " Network Training")

color = 'tab:purple'
plt.plot(loss_values, color=color)
fig.savefig("./outputs/history_values/" + name + ".jpg" , dpi=500)

save_the_model("baseline", network)
