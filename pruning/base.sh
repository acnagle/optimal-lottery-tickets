#!/bin/bash

gpu=0
hidden_size=500

mkdir -p "./results/LeNet5"

for arch in fc2 fc4; do
    mkdir -p "./results/${arch}"
    for epochs in 30 50 100 125 150; do
        python3 main.py --gpu $gpu --lr 0.1 --epochs $epochs --arch $arch --hidden-size $hidden_size --save-results > "./results/${arch}/e${epochs}_h${hidden_size}.out"
        python3 main.py --gpu $gpu --lr 0.01 --epochs $epochs --arch "LeNet5" --save-results > "./results/LeNet5/e${epochs}.out"
    done
done
