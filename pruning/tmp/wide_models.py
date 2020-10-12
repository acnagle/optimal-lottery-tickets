import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

class WideTwoLayerFC(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, sparsity, bias):
        super(WideTwoLayerFC, self).__init__()
        self.fc1 = utils.SupermaskLinear(in_features=input_size, out_features=hidden_size, sparsity=sparsity, bias=bias)
        self.fc2 = utils.SupermaskLinear(in_features=hidden_size, out_features=num_classes, sparsity=sparsity, bias=bias)

    def forward(self, x): 
        x = x.view(x.size()[0], -1) 
        x = self.fc1(x)
        x = F.relu(x) 
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)

        return output

class WideFourLayerFC(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, sparsity, bias):
        super(WideFourLayerFC, self).__init__()
        self.fc1 = utils.SupermaskLinear(in_features=input_size, out_features=hidden_size, sparsity=sparsity, bias=bias)
        self.fc2 = utils.SupermaskLinear(in_features=hidden_size, out_features=hidden_size, sparsity=sparsity, bias=bias)
        self.fc3 = utils.SupermaskLinear(in_features=hidden_size, out_features=hidden_size, sparsity=sparsity, bias=bias)
        self.fc4 = utils.SupermaskLinear(in_features=hidden_size, out_features=num_classes, sparsity=sparsity, bias=bias)

    def forward(self, x): 
        x = x.view(x.size()[0], -1) 
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        output = F.log_softmax(x, dim=1)

        return output

class WideFCLeNet5(nn.Module):
    def __init__(self, num_classes, sparsity, redundancy, bias, state_dict):
        super(WideLeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5), padding=2, bias=bias)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5), bias=bias)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        # calculate hidden_size so that the number of parameters in this wide LeNet5 has approximately the same number of parameters in the fc layers as our models
        in1 = 400    # input size of first fc layer of original LeNet5
        in2 = 120    # input size of second fc layer of original LeNet5
        in3 = 84     # input size of third fc layer of original LeNet5

        b = in1 + num_classes
        c = in1 * redundancy + in1 * redundancy * in2 + in2 * redundancy + in2 * redundancy * in3 + in3 * redundancy + in3 * redundancy * num_classes

        hidden_size = int(np.round((1 / 2) * (np.sqrt(b ** 2 + 4 * c) - b)))

        self.fc1 = utils.SupermaskLinear(in_features=in1, out_features=hidden_size, sparsity=sparsity, bias=bias)
        self.fc2 = utils.SupermaskLinear(in_features=hidden_size, out_features=hidden_size, sparsity=sparsity, bias=bias)
        self.fc3 = utils.SupermaskLinear(in_features=hidden_size, out_features=num_classes, sparsity=sparsity, bias=bias)

        self.conv1.weight.requires_grad = False
        self.conv2.weight.requires_grad = False

        if state_dict is not None:
            self.__load_weights(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = x.view(-1, 16*5*5)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        output = F.log_softmax(x, dim=-1)

        return output

    def __load_weights(self, state_dict):
        # weights that are loaded in are assumed to be coming from the LeNet5 architecture
        state_dict.popitem()    # remove fully connected layer weights
        state_dict.popitem()
        state_dict.popitem()
        state_dict['fc1.weight'] = self.fc1.weight
        state_dict['fc1.scores'] = self.fc1.scores
        state_dict['fc2.weight'] = self.fc2.weight
        state_dict['fc2.scores'] = self.fc2.scores
        state_dict['fc3.weight'] = self.fc3.weight
        state_dict['fc3.scores'] = self.fc3.scores

        self.load_state_dict(state_dict)


