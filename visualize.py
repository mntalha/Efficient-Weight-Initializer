#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 22:16:46 2023

@author: 
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

class_names = ["Alexnet","VGG-19","ResNet-50",
               "Base","GoogleNet",
               "SqueezeNet","Nvidia-Mit"]

#define img_path
result_path = "./outputs/training_results/"

def get_pretrain_results(model, dataset):  

    if model == "Alexnet":
        dataset_name = "combined_" + 'alexnet' + dataset + "Pre-training_model_prf_results.pickle"
    elif model == "VGG-19":
        dataset_name = "combined_" + 'vgg19' + dataset + "Pre-training_model_prf_results.pickle"
    elif model == "ResNet-50":
        dataset_name = "combined_" + 'resnet50' + dataset + "Pre-training_model_prf_results.pickle"
    elif model == "Base":
        dataset_name = "combined_" + 'base' + dataset + "Pre-training_model_prf_results.pickle"
    elif model == "GoogleNet":
        dataset_name = "combined_" + 'googleNet' + dataset + "Pre-training_model_prf_results.pickle"
    elif model == "SqueezeNet":
        dataset_name = "combined_" + 'squeezenet' + dataset + "Pre-training_model_prf_results.pickle"
    elif model == "Nvidia-Mit":
        dataset_name = "combined_" + 'nvidia' + dataset + "Pre-training_model_prf_results.pickle"
    else:
        print("Model name is not found")
    print(model, dataset)
    dataset_names = os.path.join(result_path,dataset_name)
    with open(dataset_names, 'rb') as file:
        results = pickle.load(file)

    return results["loss_values"]

def get_train_results(model, dataset):  

    if model == "Alexnet":
        dataset_name = 'alexnet' + "model_prf_results1" + dataset + ".pickle"
    elif model == "VGG-19":
        dataset_name = 'vgg19' + "model_prf_results1" + dataset + ".pickle"
    elif model == "ResNet-50":
        dataset_name = 'resnet50' + "model_prf_results1" + dataset + ".pickle"
    elif model == "Base":
        dataset_name = 'base' + "model_prf_results1" + dataset + ".pickle"
    elif model == "GoogleNet":
        dataset_name = 'googleNet' + "model_prf_results1" + dataset + ".pickle"
    elif model == "SqueezeNet":
        dataset_name = 'squeezenet' + "model_prf_results1" + dataset + ".pickle"
    elif model == "Nvidia-Mit":
        dataset_name = 'nvidia' + "model_prf_results1" + dataset + ".pickle"
    else:
        print("Model name is not found")
    print(dataset_name)
    dataset_names = os.path.join(result_path,dataset_name)
    with open(dataset_names, 'rb') as file:
        results = pickle.load(file)

    return results["accuracy_values"]["train_every_epoch"] #loss_values

fig, ax1 = plt.subplots(figsize=(10, 6), dpi=500)
plt.title('Training Accuracy Graph of the Models (CIFAR-10) (With Sup-Walsh)') #FashionMNIST, CIFAR-10 MNIST
for clss in class_names:
    plt.plot(get_train_results(clss, "cifar10"))

plt.xlabel("Epochs")
plt.ylabel("Accuracy Value")
plt.legend(class_names,loc = 2)
fig.savefig("CIFAR-10" + ".jpg" , dpi=750)


