#!/bin/bash

redundancy=5
device=0
lr=0.01

mkdir -p "redlenet5"
mkdir -p "widelenet5"

#for epochs in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
for epochs in 125 150; do
    for sparsity in 0 0.01 0.02 0.025 0.05 0.1 0.25 0.5 0.75 0.9 0.95 0.975 0.98 0.99; do
        python3 ../main.py --device $device --lr $lr --epochs $epochs --model "redlenet5" --load-weights "./paper/lenet5/lenet5_e50_h500.pt" --r $redundancy --sparsity $sparsity --use-relu --save-results > "./redlenet5/r${redundancy}_s${sparsity}_e${epochs}_relu.out"
        python3 ../main.py --device $device --lr $lr --epochs $epochs --model "redlenet5" --load-weights "./paper/lenet5/lenet5_e50_h500.pt" --r $redundancy --sparsity $sparsity --save-results > "./redlenet5/r${redundancy}_s${sparsity}_e${epochs}.out"
        
        python3 ../main.py --device $device --lr $lr --epochs $epochs --model "widelenet5" --load-weights "./paper/lenet5/lenet5_e50_h500.pt" --r $redundancy --sparsity $sparsity --save-results > "./widelenet5/r${redundancy}_s${sparsity}_e${epochs}.out"
    done
done

