import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torchvision.transforms as transforms

import numpy as np

import copy
import sys
import argparse
import time

import utils
import models

import gurobipy as gp
from gurobipy import GRB

def main():
    parser = argparse.ArgumentParser(description='Subset sum approximation of FC network on MNIST')

    # arguments for training/testing
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=0.0005, metavar='M',
                        help='weight decay (default: 0.0005)')
    parser.add_argument('--hidden-size', type=int, default=500, metavar='H',
                        help='number of nodes in the hidden layer of the fully connected network')
    parser.add_argument('--model', type=str, default='fc2',
                        help='model architecture to be approximated; choices include "fc2" and "fc4" for two- and four-layer fully connected networks')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--device', type=int, default=None, metavar='D',
                        help='override the default choice for a CUDA-enabled GPU by specifying the GPU\'s integer index (i.e. "0" for "cuda:0")')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', type=str, default=None,
                        help='name of the model weights to be saved')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='the number of batches to wait before logging training status')
    parser.add_argument('--data', type=str, default='../data',
                        help='location to store data')

    # arguments for target network approximation via subset sum
    parser.add_argument('--check_w_lt_eps', action='store_true', default=False,
                        help='before running subset sum, check if the weight magnitude is less than epsilon. If it is, then the approximation is thresholded to 0')
    parser.add_argument('--epsilon', type=float, default=0.01, metavar='E',
                        help='tolerance for each weight approximation (default: 0.01)')
    parser.add_argument('--c', type=float, default=1, metavar='C',
                        help='multiplicative constant; only used when deterministic approximation is not used. n = round(c * log(1/epsilon)) (default: 1)')
    parser.add_argument('--target-net', type=str, default=None,
                        help='directory path to the target network\'s pretrained weights; if None, a new network will be trained using the arguments above, and then this network will be approximated')

    args = parser.parse_args()
    print(args, '\n')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # define constants
    input_size = 784
    num_classes = 10

    # set training device
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if args.device is not None:
        device = 'cuda:' + str(args.device)
    else:
        device = torch.device('cuda' if use_cuda else 'cpu')

    # get MNIST dataset
    train_dataset = torchvision.datasets.MNIST(
        root=args.data, 
        train=True, 
        transform=transforms.ToTensor(),  
        download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root=args.data, 
        train=False, 
        transform=transforms.ToTensor()
    )

    # create data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False
    )

    # define model
    if args.model == 'fc2':
        model = models.TwoLayerFC(input_size, args.hidden_size, num_classes).to(device)
    elif args.model == 'fc4':
        model = models.FourLayerFC(input_size, args.hidden_size, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    
    if args.target_net is None:         # train a target network if one wasn't provided
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

        start = time.time()    # time to train

        test_acc_arr = np.array([np.nan] * args.epochs)

        # train model
        for epoch in range(args.epochs):
            utils.train(model, device, train_loader, optimizer, criterion, epoch+1, args.log_interval)
            test_acc_arr[epoch] = utils.test(model, device, criterion, test_loader, args.batch_size)
            scheduler.step()

        end = time.time()
        print('\nTarget network training time: {:0.2f} minutes\n'.format((end - start) / 60))
       
        target_test_acc = test_acc_arr[-1]      # final network target accuracy

        # save the model
        if args.save_model is not None:
            torch.save(model.state_dict(), args.save_model+'.pt')
    else:
        model.load_state_dict(torch.load(args.target_net, map_location=device))
        
        # inference pretrained target network on test set
        target_test_acc = utils.test(model, device, criterion, test_loader, args.batch_size)

    approx_model_dict = copy.deepcopy(model.state_dict())
    keys = list(approx_model_dict.keys())

    # determine the bounds of the uniform distribution from which the a_i coefficients are drawn
    high = 0
    low = 0
    for key, value in model.state_dict().items():
        max_val = torch.max(value.cpu()).item()
        min_val = torch.min(value.cpu()).item()
        if max_val > high:
            high = max_val
        if min_val < low:
            low = min_val

    print('The target network\'s weights are bounded from ['+str(low)+', '+str(high)+']. The a_i coefficients will be drawn uniformly from this range.\n')

    n = int(args.c * np.round(np.log2(1 / args.epsilon)))    # number of a_i coefficients to approximate each weight in the target network

    num_weights_target_total = sum(x.numel() for x in model.parameters() if x.requires_grad)
    num_weights_approx_total = num_weights_target_total * n   # total number of weights in approximated network (i.e. total number of a_i coefficients used in the approximation of the target network)
    num_weights_approx_remaining = 0    # number of weights remaining in network that approximates the target network (i.e. the number of a_i coefficients used to approximate all weights in target network)

    start = time.time()    # time to derive approximation

    for h in keys:
        print('Approximating layer '+h+' ...')
        for i in range(len(approx_model_dict[h])):
            w = approx_model_dict[h][i].cpu().data.numpy()
            a = np.random.uniform(low=low, high=high, size=n).tolist()    # every w is approximated with a unique set of random values
            if len(approx_model_dict[h][i].size()) == 0:
                w_approx, num_used = utils.FindSubset(w, a, args.epsilon, n, check_w_lt_eps=args.check_w_lt_eps)
                approx_model_dict[h][i] = w_approx
                num_weights_approx_remaining += num_used
            else:
                for j in range(len(w)):
                    w_approx, num_used = utils.FindSubset(w[j], a, args.epsilon, n, check_w_lt_eps=args.check_w_lt_eps)
                    approx_model_dict[h][i][j] = w_approx
                    num_weights_approx_remaining += num_used

    end = time.time()

    print('\nOriginal weights:')
    print(model.state_dict())
    print('\nApproximated weights:')
    print(approx_model_dict)

    # check if all weights fall within epsilon error
    weights_within_eps = True

    for key, value in model.state_dict().items():
        diff = torch.abs(model.state_dict()[key] - approx_model_dict[key])
        error_above_eps = diff > args.epsilon
        if error_above_eps.any():
            weights_within_eps = False
            
    print('\nTime to obtain approximated network using subset sum: {:.2f} minutes'.format((end - start) / 60))
    print('\nAll weights fall within eps='+str(args.epsilon)+' error:', weights_within_eps)

    if args.model == 'fc2':
        approx_model = models.TwoLayerFC(input_size, args.hidden_size, num_classes).to(device)
    elif args.model == 'fc4':
        approx_model = models.FourLayerFC(input_size, args.hidden_size, num_classes).to(device)

    approx_model.load_state_dict(approx_model_dict)

    # inference approximated target network on test set
    approx_test_acc = utils.test(approx_model, device, criterion, test_loader, args.batch_size)

    print('Accuracy of target network on the 10000 test images: {} %'.format(target_test_acc))
    print('Accuracy of approximated network on the 10000 test images: {} %'.format(approx_test_acc))

    print('\nTotal number of weights in target network:', num_weights_target_total)
    print('Total number of weights in approximated network:', num_weights_approx_total)
    print('Number of weights remaining in approximated network:', num_weights_approx_remaining)

    print('\nThe approximated network started with {:.2f} times as many weights as the target network, and ended with {:.2f} times as many weights'.format(
        num_weights_approx_total / num_weights_target_total, num_weights_approx_remaining / num_weights_target_total))

if __name__ == '__main__':
    main()
