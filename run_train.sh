#!/bin/sh
model="base alexnet resnet50 vgg19 googleNet squeezenet nvidia"
dataset="mnist cifar10 fashionmnist"
pretrained="0 1"
for mdls in $model;
do
    for data in $dataset;
    do
        for xx in $pretrained;
        do
            python main.py --model_name $mdls --pre_trained $xx --dataset $data
        done
    done
done

