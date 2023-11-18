#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 22:38:56 2023

@author: 
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
from utils import img_size
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc0 = nn.Linear(img_size, 640)
        self.fc1 = nn.Linear(640, 320)
        self.fc2 = nn.Linear(320, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, img_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x,-1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_the_model(name):
    
    path = "./outputs/saved_model/"+name+".pickle"
    
    with open(path, 'rb') as file:
        model = pickle.load(file)
    
    return model

def get_trained_model(model_name, saved = False):
    
    model = None

    if  (model_name == "alexnetmnist" or model_name == "alexnetfashionmnist")  and saved == False:
        model = models.alexnet(pretrained=False)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=(5, 5), stride=(4, 4), padding=(2, 2))
        model.classifier[6] = nn.Linear(in_features=4096, out_features= num_clss, bias=True)

    elif  model_name == "alexnetcifar10"  and saved == False:
        model = models.alexnet(pretrained=False)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=(5, 5), stride=(4, 4), padding=(2, 2))
        model.classifier[6] = nn.Linear(in_features=4096, out_features= num_clss, bias=True)

    elif model_name == "resnet34" and saved == False:
        model = models.resnet34(pretrained=False)
        model.conv1=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc =nn.Linear(in_features=512, out_features=num_clss, bias=True)
    
    elif model_name == "vgg16" and saved == False:
        model = models.vgg16(pretrained=False)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        model.classifier[6] = nn.Linear(in_features=4096, out_features=num_clss, bias=True)

    elif model_name == "mobilnet" and saved == False:
        model = models.mobilenet_v2(pretrained=False)
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.classifier[1] = nn.Linear(in_features=1280, out_features=num_clss, bias=True)
    
    elif model_name == "inception" and saved == False:
        model = models.inception_v3(pretrained=False)
        model.Conv2d_1a_3x3.conv = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        model.fc = nn.Linear(in_features=2048, out_features=num_clss, bias=True)   
    

    elif (model_name == "basemnist" or model_name == "basecifar10" or model_name == "basefashionmnist") and saved == False :
        model = Net()
    elif model_name == "basemnist" and saved == True:
        model = load_the_model("basemnist")    
    elif model_name == "basecifar10" and saved == True:
        model = load_the_model("basecifar10")  
    elif model_name == "basefashionmnist" and saved == True:
        model = load_the_model("basefashionmnist")  
    elif model_name == "combined_base" and saved == False:
        walsh = TS_WALSH()
        dnn = Net()
        model = CombinedModel("combined_base", walsh, dnn) 
        

    elif model_name == "alexnet" and saved == True:
        model = load_the_model("alexnet")   
    elif model_name == "alexnetmnist" and saved == True:
        model = load_the_model("alexnetmnist")    
    elif model_name == "alexnetcifar10" and saved == True:
        model = load_the_model("alexnetcifar10")  
    elif model_name == "alexnetfashionmnist" and saved == True:
        model = load_the_model("alexnetfashionmnist")   
    elif model_name == "combined_alexnet" and saved == False:
        walsh = TS_WALSH()
        dnn = models.alexnet(pretrained=False)
        dnn.features[0] = nn.Conv2d(1, 64, kernel_size=(5, 5), stride=(4, 4), padding=(2, 2))
        dnn.classifier[6] = nn.Linear(in_features=4096, out_features= num_clss, bias=True)
        model = CombinedModel("combined_alexnet", walsh, dnn) 
     
    elif (model_name == "resnet50mnist" or model_name == "resnet50fashionmnist") and saved == False:
        model = models.resnet50(pretrained=False)
        model.conv1=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc =nn.Linear(in_features=2048, out_features=num_clss, bias=True)
    elif model_name == "resnet50cifar10"  and saved == False:
        model = models.resnet50(pretrained=False)
        model.conv1=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc =nn.Linear(in_features=2048, out_features=num_clss, bias=True)   

    elif model_name == "resnet50" and saved == True:
        model = load_the_model("resnet50")      
    elif model_name == "resnet50mnist" and saved == True:
        model = load_the_model("resnet50mnist")    
    elif model_name == "resnet50cifar10" and saved == True:
        model = load_the_model("resnet50cifar10")  
    elif model_name == "resnet50fashionmnist" and saved == True:
        model = load_the_model("resnet50fashionmnist")  
    elif model_name == "combined_resnet50" and saved == False:
        walsh = TS_WALSH()
        dnn = models.resnet50(pretrained=False)
        dnn.conv1=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        dnn.fc =nn.Linear(in_features=2048, out_features=num_clss, bias=True)
        model = CombinedModel("combined_resnet50", walsh, dnn) 

    elif(model_name == "vgg19mnist" or model_name == "vgg19fashionmnist") and saved == False:
        model = models.vgg19(pretrained=False)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        model.classifier[6] = nn.Linear(in_features=4096, out_features=num_clss, bias=True)
    elif model_name == "vgg19cifar10" and saved == False:
        model = models.vgg19(pretrained=False)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        model.classifier[6] = nn.Linear(in_features=4096, out_features=num_clss, bias=True)
    elif model_name == "vgg19" and saved == True:
        model = load_the_model("vgg19")      
    elif model_name == "vgg19mnist" and saved == True:
        model = load_the_model("vgg19mnist")    
    elif model_name == "vgg19cifar10" and saved == True:
        model = load_the_model("vgg19cifar10")  
    elif model_name == "vgg19fashionmnist" and saved == True:
        model = load_the_model("vgg19fashionmnist")  
    elif model_name == "combined_vgg19" and saved == False:
        walsh = TS_WALSH()        
        dnn = models.vgg19(pretrained=False)
        dnn.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        dnn.classifier[6] = nn.Linear(in_features=4096, out_features=num_clss, bias=True)
        model = CombinedModel("combined_resnet50", walsh, dnn) 

    elif  model_name == "googleNetcifar10" and saved == False:
        model = models.googlenet(pretrained=False,transform_input=False)
        model.conv1.conv = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(in_features=1024, out_features=num_clss, bias=True)
    elif (model_name == "googleNetmnist" or model_name == "googleNetfashionmnist") and saved == False:
        model = models.googlenet(pretrained=False,transform_input=False)
        model.conv1.conv = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(in_features=1024, out_features=num_clss, bias=True)
    elif model_name == "googleNet" and saved == True:
        model = load_the_model("googleNet")   
    elif model_name == "googleNetmnist" and saved == True:
        model = load_the_model("googleNetmnist")    
    elif model_name == "googleNetcifar10" and saved == True:
        model = load_the_model("googleNetcifar10")  
    elif model_name == "googleNetfashionmnist" and saved == True:
        model = load_the_model("googleNetfashionmnist")    
    elif model_name == "combined_googleNet" and saved == False:
        walsh = TS_WALSH()        
        dnn = models.googlenet(pretrained=False,transform_input=False)
        dnn.conv1.conv = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        dnn.fc = nn.Linear(in_features=1024, out_features=num_clss, bias=True)
        model = CombinedModel("combined_googleNet", walsh, dnn) 


    elif (model_name == "squeezenetmnist" or model_name == "squeezenetfashionmnist") and saved == False:
        model = models.squeezenet1_0(pretrained=False)
        model.features[0] = nn.Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2))
        model.classifier[1] = nn.Conv2d(512, num_clss, kernel_size=(1, 1), stride=(1, 1))
    elif model_name == "squeezenetcifar10" and saved == False:
        model = models.squeezenet1_0(pretrained=False)
        model.features[0] = nn.Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2))
        model.classifier[1] = nn.Conv2d(512, num_clss, kernel_size=(1, 1), stride=(1, 1))
    elif model_name == "squeezenet" and saved == True:
        model = load_the_model("squeezenet")  
    elif model_name == "squeezenetmnist" and saved == True:
        model = load_the_model("squeezenetmnist")    
    elif model_name == "squeezenetcifar10" and saved == True:
        model = load_the_model("squeezenetcifar10")  
    elif model_name == "squeezenetfashionmnist" and saved == True:
        model = load_the_model("squeezenetfashionmnist")  
    elif model_name == "combined_squeezenet" and saved == False:
        walsh = TS_WALSH()        
        dnn = models.squeezenet1_0(pretrained=False)
        dnn.features[0] = nn.Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2))
        dnn.classifier[1] = nn.Conv2d(512, num_clss, kernel_size=(1, 1), stride=(1, 1))
        model = CombinedModel("combined_squeezenet", walsh, dnn) 
    elif (model_name == "nvidiamnist" or model_name == "nvidiafashionmnist") and saved == False:
        model = SegformerForImageClassification.from_pretrained("nvidia/mit-b0")
        model.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(1, 32, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
        model.classifier = nn.Linear(in_features=256, out_features=num_clss, bias=True)

    elif model_name == "nvidiacifar10" and saved == False:
        model = SegformerForImageClassification.from_pretrained("nvidia/mit-b0")
        model.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(1, 32, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
        model.classifier = nn.Linear(in_features=256, out_features=num_clss, bias=True)

    elif model_name == "nvidia" and saved == True:
        model = load_the_model("nvidia")   
    elif model_name == "nvidiamnist" and saved == True:
        model = load_the_model("nvidiamnist")    
    elif model_name == "nvidiacifar10" and saved == True:
        model = load_the_model("nvidiacifar10")  
    elif model_name == "nvidiafashionmnist" and saved == True:
        model = load_the_model("nvidiafashionmnist")  
    elif model_name == "combined_nvidia" and saved == False:
        walsh = TS_WALSH()        
        dnn = SegformerForImageClassification.from_pretrained("nvidia/mit-b0")
        dnn.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(1, 32, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
        dnn.classifier = nn.Linear(in_features=256, out_features=num_clss, bias=True)
        model = CombinedModel("combined_nvidia", walsh, dnn) 

    return model




