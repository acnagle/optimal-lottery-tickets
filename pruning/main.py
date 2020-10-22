from __future__ import print_function
import os
import math
import random
import numpy as np
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.autograd as autograd

from utils import data
from utils.train_test import train, test
import models

from args import args

def main():
    print(args, '\n')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = get_device(args)
    data = get_dataset(args)
    model = get_model(args, data, device)
    
    print('\n'+str(model)+'\n')
  
    # Only pass the parameters where p.requires_grad == True to the optimizer
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd,
    )
    
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    test_acc_arr = np.array([np.nan] * args.epochs)

    start = time.time()
    for epoch in range(args.epochs):
        train(model, device, data.train_loader, optimizer, criterion, epoch+1, args.log_interval)
        test_acc_arr[epoch] = test(model, device, data.test_loader, criterion, args.batch_size)
        scheduler.step()
    end = time.time()
    total_time = (end - start) / 60

    num_weights = sum(x.numel() for x in model.parameters() if x.requires_grad)     # This calcuation does not include the number of weights in convolutional layers, including baseline models, since we are interested in observing the number of parameters in fully connected layers only. Convolutional layers are randomly initialized and never trained/pruned in any of the models. Note that num_weights is equal to the number of parameters that are being updated in the network

    print('\nTotal time spent pruning/training: {:.2f} minutes'.format(total_time))
    print('Total number of parameters in model:', num_weights)

    if args.arch not in ['TwoLayerFC', 'FourLayerFC', 'LeNet5']:
        num_params_pruned = int(num_weights * args.sparsity)
        num_params_remaining = num_weights - num_params_pruned

        print('Number of parameters in pruned model:', num_params_remaining)
    else:
        num_params_remaining = None

    if args.save_results or args.save_model:
        save(model, test_acc_arr, total_time, num_weights, num_params_remaining, args)


def get_device(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if args.gpu is None:
        device = torch.device('cuda:0' if use_cuda else 'cpu')
    else:
        device = 'cuda:' + str(args.gpu)
  
    if use_cuda:
        torch.cuda.device(device)

    print('Using device {} for training and testing'.format(device))

    return device


def get_dataset(args):
    print('Benchmarking with the {} dataset'.format(args.dataset))
    dataset = getattr(data, args.dataset.upper())(args)
    
    return dataset


def get_model(args, data, device):
    if args.redundancy <= 0:
        raise ValueError('Redundancy factor must be greater than or equal to 1')
    print('Creating model {}'.format(args.arch))
    model = models.__dict__[args.arch](data.INPUT_SIZE, data.NUM_CLASSES, args)
    
    if not args.no_cuda:
        model.cuda(device)
    
    if args.freeze_weights:
        freeze_model_weights(model)
    
    return model


def freeze_model_weights(model):
    print('\nFreezing model weights:')

    for weight_attr in ['weight', 'weight1', 'weight2']:
        for n, m in model.named_modules():
            if hasattr(m, weight_attr) and getattr(m, weight_attr) is not None:
                print(f'  No gradient to {n}.{weight_attr}')
                getattr(m, weight_attr).requires_grad = False
                if getattr(m, weight_attr).grad is not None:
                    print(f'  Setting gradient of {n}.{weight_attr} to None')
                    getattr(m, weight_attr).grad = None
            
                if hasattr(m, "bias") and m.bias is not None:
                    print(f'  No gradient to {n}.bias')
                    m.bias.requires_grad = False
                    if m.bias.grad is not None:
                        print(f'  Setting gradient of {n}.bias to None')
                        m.bias.grad = None


def save(model, test_acc_arr, total_time, num_weights, num_params_remaining, args): 
    if args.arch not in ['TwoLayerFC', 'FourLayerFC', 'LeNet5']:
        filename = 'r'+str(args.redundancy)+'_s'+str(args.sparsity)+'_'
    else:
        filename = ''

    filename += 'e'+str(args.epochs)+'_h'+str(args.hidden_size)

    if args.use_relu:
        filename += '_relu'

    if args.save_results:
        save_dir = './results/'+args.arch
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        np.savez(save_dir+'/'+filename+'.npz',
            args=vars(args),
            test_acc=test_acc_arr,
            total_time=total_time,
            sparsity=args.sparsity,
            num_weights=num_weights,
            num_params_remaining=num_params_remaining
        )

    if args.save_model:
        save_dir = './weights/'+args.arch
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(model.state_dict(), save_dir+'/'+filename+'.pt')


if __name__ == '__main__':
    main()
