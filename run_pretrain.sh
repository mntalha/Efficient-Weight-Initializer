#!/bin/sh
model="base alexnet resnet50 vgg19 googleNet squeezenet nvidia";
dataset="mnist cifar10 fashionmnist country211"

for mdls in $model;
do
    for data in $dataset:
    do
        python ts_walsh_training.py --model_name $mdls --dataset $data
    done
done
