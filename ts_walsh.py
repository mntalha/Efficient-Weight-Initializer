#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 22:37:13 2023

@author: talha
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

num_clss = 10 

#Walsh 
zero  = torch.tensor([1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1]).view(1,16)
one   = torch.tensor([1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1]).view(1,16)
two   = torch.tensor([1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1]).view(1,16)
three = torch.tensor([1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1]).view(1,16)
four  = torch.tensor([1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1]).view(1,16)
five  = torch.tensor([1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1]).view(1,16)
six   = torch.tensor([1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1]).view(1,16)
seven = torch.tensor([1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1]).view(1,16)
eight = torch.tensor([1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,-1,1,-1,1]).view(1,16)
nine  = torch.tensor([1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1]).view(1,16)

walsh = torch.stack([zero, one, two, three, four, five, six, seven, eight, nine])
targets = torch.tensor([i for i in range(num_clss)])


#Sampling

# Dataloader
from Dataset import call_dataloader

trainloader, test_loader = call_dataloader("./data", batch_size = 100, name = "mnist")
sample = next(iter(trainloader))

sampled_data = []

for idx in range(num_clss):
    indexes = torch.where(sample[1] == idx)[0]
    size = indexes.size(0)
    random_index = torch.randint(0, size, (1,)).item()
    random_element = sample[0][indexes[random_index]].view(-1)
    sampled_data.append(np.array(random_element))



from sklearn.manifold import TSNE
x_tsne = TSNE(n_components = 2, perplexity = 5, n_jobs=-1).fit_transform(np.array(sampled_data))
x_tsne = torch.tensor(x_tsne)

def get_values():
    global x_tsne, targets
    return x_tsne, targets

# import seaborn as sns
# import matplotlib.pyplot as plt
# plt.figure(figsize=(16,10))
# sns.scatterplot(
#     x=x_tsne[:,0], y=x_tsne[:,1],
#     hue=targets,
#     palette=sns.color_palette("hls", 10),
#     legend="full",
#     alpha=1,
#     s=300
# )

#TS_WALSH NETWORK 
from utils import img_size

class TS_WALSH(nn.Module):
    def __init__(self, size = img_size):
        super(TS_WALSH, self).__init__()
        
        self.prebn = nn.BatchNorm1d(2)

        self.tse = nn.Linear(2,4)
        
        self.bn = nn.BatchNorm1d(4)

        self.tse1 = nn.Linear(4,8)

        self.bn1 = nn.BatchNorm1d(8)

        self.tse2 = nn.Linear(8,16)

        self.bn2 = nn.BatchNorm1d(16)
        
        self.tse3 = nn.Linear(16, size)

        self.bn3 = nn.BatchNorm1d(size)

    def forward(self, x, target):
        
        if x.shape[1] != 2: 
            raise Exception("Input size is not as expected", x.shape)  
        x = (self.tse((x)))
        x = (self.tse1(x))
        x = (self.tse2(x))
        global walsh 
        x += walsh[target].squeeze() 
        x =self.tse3((x))
        
        return (x.unsqueeze(1).unsqueeze(1))


class CombinedModel(nn.Module):
  def __init__(self, name, tswalsh, dnn):
    super(CombinedModel, self).__init__()
    self.tswaslh = tswalsh
    self.dnn = dnn
    self.name = name
    
  def forward(self,  x, target):
    x = self.tswaslh(x,target).view(x.shape[0], 1, int(np.sqrt(img_size)), int(np.sqrt(img_size)) )
    if self.name == "nvidia":
        pass
    x = self.dnn(x)
    return x


# network = TS_WALSH()
# result = network(x_tsne,targets)
