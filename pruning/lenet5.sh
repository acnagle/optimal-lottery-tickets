#!/bin/bash

gpu=0
redundancy=5
lr=0.01

mkdir -p "./results/RedLeNet5"
mkdir -p "./results/WideLeNet5"

for epochs in 30 50 100 125 150; do
    for sparsity in 0 0.01 0.02 0.025 0.05 0.1 0.25 0.5 0.75 0.9 0.95 0.975 0.98 0.99; do
        python3 main.py --gpu $gpu --lr $lr --epochs $epochs --arch "RedLeNet5" --load-weights "./weights/LeNet5/LeNet5_e50_h500.pt" --redundancy $redundancy --sparsity $sparsity --freeze-weights --use-relu --save-results > "./results/RedLeNet5/r${redundancy}_s${sparsity}_e${epochs}_relu.out"
        python3 main.py --gpu $gpu --lr $lr --epochs $epochs --arch "RedLeNet5" --load-weights "./weights/LeNet5/LeNet5_e50_h500.pt" --redundancy $redundancy --sparsity $sparsity --freeze-weights --save-results > "./results/RedLeNet5/r${redundancy}_s${sparsity}_e${epochs}.out"
        
        python3 main.py --gpu $gpu --lr $lr --epochs $epochs --arch "WideLeNet5" --load-weights "./weights/LeNet5/LeNet5_e50_h500.pt" --redundancy $redundancy --sparsity $sparsity --freeze-weights --save-results > "./results/WideLeNet5/r${redundancy}_s${sparsity}_e${epochs}.out"
    done
done

