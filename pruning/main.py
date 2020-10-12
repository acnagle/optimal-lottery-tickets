from __future__ import print_function
import argparse
import os
import math
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

import redundant_models as rm
import baseline_models as bm
import wide_models as wm
import utils
from utils import SupermaskLinear

def main():
    parser = argparse.ArgumentParser(description='MNIST Pruning using Modified FC Network')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=0.0005, metavar='M',
                    help='weight decay (default: 0.0005)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
    parser.add_argument('--device', type=int, default=None, metavar='D',
                    help='override the default choice for a CUDA-enabled GPU by specifying the GPU\'s integer index (i.e. "0" for "cuda:0")')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
    parser.add_argument('--save-model', type=str, default=None,
                    help='save the trained model; provide a path and filename to save the weights')
    parser.add_argument('--save-results', action='store_true', default=False,
                    help='save the results and arguments of this experiment')
    parser.add_argument('--hidden-size', type=int, default=500, metavar='H',
                    help='number of nodes in the FC layers of the network. Important: when the model of choice is a wide network, the number of hidden nodes is determined based on this argument and the --r argument. Recalculating the number of hidden nodes allowed us to determine how many hidden nodes the wide networks should have in order to have the same number of parameters as a network with our structure.')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='the number of batches to wait before logging training status')
    parser.add_argument('--data', type=str, default='../data',
                    help='directory of dataset')
    parser.add_argument('--model', type=str, default='redfc2',
            help='name of model to train. can be "fc2", "fc4", "redfc2", "redfc4", "widefc2", "widefc4", "lenet5", "redlenet5", "widelenet5" (default: "redfc2")')    # TODO: Add resnet models to list, update this list to use the full names of each model as specified in redundant.py, baseline.py and wide.py. can I print the names from 'all' vector in __init__.py from models directory?
    parser.add_argument('--load-weights', type=str, default=None,
                    help='provide a path to load the weights from; only used if --model is "redlenet5")')
    parser.add_argument('--use-relu', action='store_true', default=False,
                    help='if true, relu activation will be included when adding redundancy')
    parser.add_argument('--sparsity', type=float, default=0.5, metavar='S',
                    help='the ratio of weights to remove in each layer (default: 0.5)')
    parser.add_argument('--r', type=int, default=5, metavar='R',
                    help='number of units of redundancy (default: 5)')
    parser.add_argument('--bias', action='store_true', default=None,
                    help='boolean flag to indicate the inclusion of bias terms in the neural network (default: None)')
    parser.add_argument('--dataset', type=str, default='MNIST',
                    help='dataset to train the model with. Can be CIFAR10 or MNIST (default: MNIST)')
    parser.add_argument('--num-workers', type=int, default=4, metavar='M',
                    help='number of workers')

    args = parser.parse_args()
    print(args, '\n')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # set training device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    
    if args.device is not None:
        device = 'cuda:' + str(args.device)
    else:
        device = torch.device('cuda' if use_cuda else 'cpu')

    # get dataset
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}
    
    # TODO: insert get_dataset method
    

    if args.load_weights is not None:
        state_dict = torch.load(args.load_weights)
    else:
        state_dict = None

    if args.model == 'redfc2':
        print('Pruning a Two-Layer Fully Connected Redundant Network ...')
        model = rm.RedTwoLayerFC(input_size, args.hidden_size, num_classes, args.sparsity, args.r, args.bias, args.use_relu).to(device)
    elif args.model == 'redfc4':
        print('Pruning a Four-Layer Fully Connected Redundant Network ...')
        model = rm.RedFourLayerFC(input_size, args.hidden_size, num_classes, args.sparsity, args.r, args.bias, args.use_relu).to(device)
    elif args.model == 'redlenet5':
        print('Training a RedLeNet5 network ...')
        model = rm.RedLeNet5(num_classes, args.sparsity, args.r, args.bias, args.use_relu, state_dict).to(device)
    elif args.model == 'redvgg16':
        print('Training a RedVGG16 network ...')
        model = rm.RedVGG('VGG16', num_classes, args.sparsity, args.r, args.bias, args.use_relu, state_dict).to(device)
    elif args.model == 'widefc2':
        print('Pruning a Wide Two-Layer Fully Connected network ...')
        # calculate size of hidden layers
        a = sum(x.numel() for x in rm.RedTwoLayerFC(input_size, args.hidden_size, num_classes, args.sparsity, args.r, args.bias, args.use_relu).parameters() if x.requires_grad)
        b = input_size + num_classes
        hidden_size = int(np.round(a / b))
        model = wm.WideTwoLayerFC(input_size, hidden_size, num_classes, args.sparsity, args.bias).to(device)
    elif args.model == 'widefc4':
        print('Pruning a Wide Four-Layer Fully Connected network ...')
        # calculate size of hidden layers
        a = sum(x.numel() for x in rm.RedFourLayerFC(input_size, args.hidden_size, num_classes, args.sparsity, args.r, args.bias, args.use_relu).parameters() if x.requires_grad)
        b = input_size + num_classes
        hidden_size = int(np.round((1 /4) * (np.sqrt(8 * a + (b ** 2)) - b)))
        model = wm.WideFourLayerFC(input_size, hidden_size, num_classes, args.sparsity, args.bias).to(device)
    elif args.model == 'widelenet5':
        print('Pruning a Wide LeNet5 network ...')
        model = wm.WideLeNet5(num_classes, args.sparsity, args.r, args.bias, state_dict).to(device)
    elif args.model == 'fc2':
        print('Training a Two-Layer Fully Connected Network ...')
        model = bm.TwoLayerFC(input_size, args.hidden_size, num_classes, args.bias).to(device)
    elif args.model == 'fc4':
        print('Training a Four-Layer Fully Connected Network ...')
        model = bm.FourLayerFC(input_size, args.hidden_size, num_classes, args.bias).to(device)
    elif args.model == 'lenet5':
        print('Training a LeNet5 network ...')
        model = bm.LeNet5(num_classes, args.bias).to(device)
    elif args.model == 'vgg16':
        print('Training a VGG16 network ...')
        model = bm.VGG('VGG16')

    # only pass the parameters where p.requires_grad == True to the optimizer
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
        utils.train(model, device, train_loader, optimizer, criterion, epoch+1, args.log_interval)
        test_acc_arr[epoch] = utils.test(model, device, criterion, test_loader, args.batch_size)
        scheduler.step()
    end = time.time()
    train_time = (end - start) / 60     # in minutes

    if args.load_weights is not None:
        if args.model in ['redlenet5', 'widelenet5']:
            total_num_params = sum(x.numel() for x in model.parameters() if not x.requires_grad)
    else:
        total_num_params = sum(x.numel() for x in model.parameters() if x.requires_grad)   # Note that this does not include both the score and weight parameters, which would be twice this number
        if args.model in 'widelenet5':
            total_num_params = sum(x.numel() for x in model.parameters() if not x.requires_grad)

    if args.model == 'lenet5':
        total_num_params = sum(x.numel() for x in model.parameters())

    print(args)
    print('\n\nTotal time spent pruning/training: {:.2f} minutes'.format(train_time))
    print('Total number of parameters in model:', total_num_params)
    
    if args.model in ['redfc2', 'redfc4', 'redlenet5', 'widefc2', 'widefc4', 'widelenet5']:
        if args.model in ['redlenet5', 'widelenet5']:
            param_count = 0
            for name, param in model.named_parameters():
                if name in ['red1.scores1', 'red1.scores2', 'red2.scores1', 'red2.scores2', 'red3.scores1', 'red3.scores2', 'fc1.scores', 'fc2.scores', 'fc3.scores']:
                    param_count += param.numel()

            params_pruned = int(param_count * args.sparsity)
            pruned_num_params_remaining = total_num_params - params_pruned
        else:
            pruned_num_params_remaining = int(total_num_params * (1-args.sparsity))    
    

        print('Number of parameters in pruned model:', pruned_num_params_remaining)

    if args.save_results:
        if not os.path.exists(args.model):
            os.makedirs(args.model)

        if args.model in ['redfc2', 'redfc4', 'redlenet5', 'widefc2', 'widefc4', 'widelenet5']:
            filename = './'+args.model+'/r'+str(args.r)+'_s'+str(args.sparsity)+'_e'+str(args.epochs)+'_h'+str(args.hidden_size)
            
            if args.use_relu:
                filename += '_relu'

            np.savez(filename+'.npz',
                args=vars(args),
                test_acc=test_acc_arr,
                train_time=train_time,
                sparsity=args.sparsity,
                num_params=total_num_params,
                num_params_remaining=pruned_num_params_remaining
            )
        else:
            filename = './'+args.model+'/'+args.model+'_e'+str(args.epochs)+'_h'+str(args.hidden_size)
            
            np.savez(filename+'.npz',
                args=vars(args),
                test_acc=test_acc_arr,
                train_time=train_time,
                sparsity=args.sparsity,
                num_params=total_num_params
            )

    if args.save_model is not None:
        torch.save(model.state_dict(), args.save_model+'.pt')


if __name__ == '__main__':
    main()
