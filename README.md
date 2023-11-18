# Sup-Walsh ("Effective Weight Initialization Technique)

We propose the auxiliary network model, called  $\textbf{Sup-Walsh}$  (Support Walsh), as a weight re-organizer that remains connected throughout the pretraining process. In this structure, our goal is to enhance the boundaries between classes, believing that expanding these boundaries makes them more distinguishable heuristically.

The proposed solution has been tested on three publicly available datasets, MNIST \cite{deng2012mnist}, CIFAR-10 \cite{recht2018cifar}, Fashion-MNIST \cite{xiao2017fashion}. Additionally, its success has been measured on popular classification models by integrating the developed support network.
Sup-Walsh consistently outperformed others in the experiments although relative improvements vary across models and datasets

# Contribtuion


This README file was generated using GitHub Copilot. It is intended for contribution purposes.


# Deep Learning Weight Initialization Techniques

This repository contains implementations of proposed weight initialization techniques used in deep learning.

## Traditional Techniques

1. **Zero Initialization** - All weights are initialized to zero.
2. **Random Initialization** - Weights are initialized randomly.
3. **He Initialization** - Weights are initialized keeping in mind the size of the previous layer which helps in attaining a global minimum of the cost function faster and more accurately.
4. **Xavier/Glorot Initialization** - It's a way of initializing the weights such that the variance remains the same for x and y.

## Requirements

- Python 3.x
- matplotlib 3.7.2
- matplotlib-inline 0.1.6
- numpy 1.24.3
- pip  23.3
- QtPy 2.2.0
- requests 2.31.0
- torch 2.1.0
- scikit-learn 1.3.0

## Usage

Each technique is implemented in its own Python file. You can use them by importing the required file into your project.



