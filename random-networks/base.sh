#!/bin/bash

device=1
hidden_size=500

mkdir -p "lenet5"

for model in fc2 fc4; do
    mkdir -p $model
#    for epochs in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    for epochs in 30 50 100 125 150; do
        python3 ../main.py --device $device --lr 0.1 --epochs $epochs --model $model --hidden-size $hidden_size --save-results > "./${model}/e${epochs}_h${hidden_size}.out"
        python3 ../main.py --device $device --lr 0.01 --epochs $epochs --model "lenet5" --save-results > "./lenet5/e${epochs}.out"
    done
done
