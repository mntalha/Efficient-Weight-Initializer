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
import random
import numpy as np

torch.manual_seed(42)
torch.cuda.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)


from models import get_trained_model
from ts_walsh import get_values
learning_rate = 0.0001
epochs = 100000


def save_the_model(name, network):
    
    path = "./outputs/saved_model/"+name+".pickle"
    
    with open(path, 'wb') as f:
        pickle.dump(network.dnn, f)

    

name = "combined_base"
network = get_trained_model(name)
criterion = torch.nn.CrossEntropyLoss()  #KLDivLoss , CrossEntropyLoss
optimizer = optim.Adam(network.parameters(), lr=learning_rate)

#initial_weights = network.dnn.fc3.weight.clone()
network.train()
x_tsne, targets = get_values()
one_hot_vectors = torch.nn.functional.one_hot(targets, 10).float()
loss_max = 9999
loss_values = []
for i in range(epochs):
        optimizer.zero_grad()
        output = network(x_tsne, targets)
        loss = criterion(output, targets)
        loss_values.append(loss.item())
        if loss_max > loss:
            loss_max = loss
            save_the_model("baseline", network)
        loss.backward()
        optimizer.step()
network.eval()

# print(all(network.dnn.fc3.weight == initial_weights))

fig, ax1 = plt.subplots(figsize=(10, 6), dpi=500)

plt.title("Support "+ name + " Network Training")

color = 'tab:purple'
plt.plot(loss_values, color=color)
fig.savefig("./outputs/history_values/" + name + ".jpg" , dpi=500)

# save_the_model("baseline", network)
