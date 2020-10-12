#!/bin/bash

redundancy=5
device=1
lr=0.1
hidden_size=500

#for epochs in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
for epochs in 125 150; do
    for model in redfc2 redfc4; do
        mkdir -p $model
#        for sparsity in 0 0.01 0.02 0.025 0.05 0.1 0.25 0.5 0.75 0.9 0.95 0.975 0.98 0.99; do
        for sparsity in 0.1 0.5 0.95; do
            python3 ../main.py --device $device --lr $lr --epochs $epochs --model $model --hidden-size $hidden_size --r $redundancy --sparsity $sparsity --use-relu --save-results > "./${model}/r${redundancy}_s${sparsity}_e${epochs}_h${hidden_size}_relu.out"
            python3 ../main.py --device $device --lr $lr --epochs $epochs --model $model --hidden-size $hidden_size --r $redundancy --sparsity $sparsity --save-results > "./${model}/r${redundancy}_s${sparsity}_e${epochs}_h${hidden_size}.out"
        done
    done
done

#for epochs in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
for epochs in 125 150; do
    for model in widefc2 widefc4; do
        mkdir -p $model
#        for sparsity in 0 0.01 0.02 0.025 0.05 0.1 0.25 0.5 0.75 0.9 0.95 0.975 0.98 0.99; do
        for sparsity in 0.1 0.5 0.95; do
            python3 ../main.py --device $device --lr $lr --epochs $epochs --model $model --hidden-size $hidden_size --r $redundancy --sparsity $sparsity --save-results > "./${model}/r${redundancy}_s${sparsity}_e${epochs}_h${hidden_size}.out"
        done
    done
done


