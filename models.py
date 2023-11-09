#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 22:38:56 2023

@author: talha
"""

from torchvision import models
import torch.nn as nn
import torch
from transformers import AutoImageProcessor, ViTForImageClassification
from transformers import SegformerFeatureExtractor, SegformerForImageClassification
from transformers import SegformerForImageClassification, SegformerConfig
import torch.nn.functional as F
from ts_walsh import CombinedModel, TS_WALSH
import pickle 
num_clss = 10 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc0 = nn.Linear(28*28, 640)
        self.fc1 = nn.Linear(640, 320)
        self.fc2 = nn.Linear(320, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_the_model(name):
    
    path = "./outputs/saved_model/"+name+".pickle"
    
    with open(path, 'rb') as file:
        model = pickle.load(file)
    
    return model

def get_trained_model(model_name, saved = False):
    
    model = None
    
    if model_name == "alexnet" and saved == False:
        model = models.alexnet(pretrained=False)
        model.features[0] = nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        model.classifier[6] = nn.Linear(in_features=4096, out_features= num_clss, bias=True)
    elif model_name == "resnet50" and saved == False:
        model = models.resnet50(pretrained=False)
        model.conv1=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc =nn.Linear(in_features=2048, out_features=num_clss, bias=True)
    elif model_name == "resnet34" and saved == False:
        model = models.resnet34(pretrained=False)
        model.conv1=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc =nn.Linear(in_features=512, out_features=num_clss, bias=True)
    elif model_name == "vgg19" and saved == False:
        model = models.vgg19(pretrained=False)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        model.classifier[6] = nn.Linear(in_features=4096, out_features=num_clss, bias=True)
    elif model_name == "vgg16" and saved == False:
        model = models.vgg16(pretrained=False)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        model.classifier[6] = nn.Linear(in_features=4096, out_features=num_clss, bias=True)
    elif model_name == "googleNet" and saved == False:
        model = models.googlenet(pretrained=False,transform_input=False)
        model.conv1.conv = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(in_features=1024, out_features=num_clss, bias=True)
    elif model_name == "mobilnet" and saved == False:
        model = models.mobilenet_v2(pretrained=False)
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.classifier[1] = nn.Linear(in_features=1280, out_features=num_clss, bias=True)
    elif model_name == "squeezenet" and saved == False:
        model = models.squeezenet1_0(pretrained=False)
        model.features[0] = nn.Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2))
        model.classifier[1] = nn.Conv2d(512, num_clss, kernel_size=(1, 1), stride=(1, 1))
    elif model_name == "inception" and saved == False:
        model = models.inception_v3(pretrained=False)
        model.Conv2d_1a_3x3.conv = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        model.fc = nn.Linear(in_features=2048, out_features=num_clss, bias=True)   
    elif model_name == "nvidia" and saved == False:
        model = SegformerForImageClassification.from_pretrained("nvidia/mit-b0")
        model.classifier = nn.Linear(in_features=256, out_features=num_clss, bias=True)
    elif model_name == "base" and saved == False :
        model = Net()
    elif model_name == "combined_base" and saved == False:
        walsh = TS_WALSH()
        dnn = Net()
        model = CombinedModel("combined_base", walsh, dnn) 
    elif model_name == "base" and saved == True:
        model = load_the_model("baseline") 
    return model