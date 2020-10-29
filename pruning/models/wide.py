import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.subnet import SupermaskLinear

class WideTwoLayerFC(nn.Module):
    def __init__(self, input_size, num_classes, args):
        super(WideTwoLayerFC, self).__init__()
        
        # Determine size of hidden layers for this wide network based on number of parameters in a redundant network with 
        # args.redundancy extra links per connection and a hidden size of args.hidden_size. The number of parameters in
        # this wide network must approximately match the number of parameters in the redundant network.
        num_params_base_net = input_size * args.hidden_size + args.hidden_size * num_classes
        num_params_red_net = num_params_base_net * args.redundancy + args.redundancy * (input_size + args.hidden_size)
        wide_hidden_size = int(round(num_params_red_net/ (input_size + num_classes)))

        self.fc1 = SupermaskLinear(in_features=input_size, out_features=wide_hidden_size, sparsity=args.sparsity, bias=args.bias)
        self.fc2 = SupermaskLinear(in_features=wide_hidden_size, out_features=num_classes, sparsity=args.sparsity, bias=args.bias)

    def forward(self, x): 
        x = x.view(x.size()[0], -1) 
        x = self.fc1(x)
        x = F.relu(x) 
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)

        return output

class WideFourLayerFC(nn.Module):
    def __init__(self, input_size, num_classes, args):
        super(WideFourLayerFC, self).__init__()
        
        # Determine size of hidden layers for this wide network based on number of parameters in a redundant network with 
        # args.redundancy extra links per connection and a hidden size of args.hidden_size. The number of parameters in
        # this wide network must approximately match the number of parameters in the redundant network.
        num_params_base_net = input_size * args.hidden_size + 2 * args.hidden_size * args.hidden_size + args.hidden_size * num_classes
        num_params_red_net = num_params_base_net * args.redundancy + args.redundancy * (input_size + 3 * args.hidden_size)
        wide_hidden_size = int(round((1 / 4) * (math.sqrt(8 * num_params_red_net + ((input_size + num_classes) ** 2)) - (input_size + num_classes))))

        self.fc1 = SupermaskLinear(in_features=input_size, out_features=wide_hidden_size, sparsity=args.sparsity, bias=args.bias)
        self.fc2 = SupermaskLinear(in_features=wide_hidden_size, out_features=wide_hidden_size, sparsity=args.sparsity, bias=args.bias)
        self.fc3 = SupermaskLinear(in_features=wide_hidden_size, out_features=wide_hidden_size, sparsity=args.sparsity, bias=args.bias)
        self.fc4 = SupermaskLinear(in_features=wide_hidden_size, out_features=num_classes, sparsity=args.sparsity, bias=args.bias)

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

class WideLeNet5(nn.Module):
    def __init__(self, input_size, num_classes, args):
        super(WideLeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5), padding=2, bias=args.bias)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5), bias=args.bias)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        # calculate hidden_size so that the number of parameters in this wide LeNet5 has approximately the same number of parameters in the fc layers as our models
        in1 = 400    # input size of first fc layer of original LeNet5
        in2 = 120    # input size of second fc layer of original LeNet5
        in3 = 84     # input size of third fc layer of original LeNet5

        b = in1 + num_classes
        c = in1 * args.redundancy + in1 * args.redundancy * in2 + in2 * args.redundancy + in2 * args.redundancy * in3 + in3 * args.redundancy + in3 * args.redundancy * num_classes

        hidden_size = int(round((1 / 2) * (math.sqrt(b ** 2 + 4 * c) - b)))

        self.fc1 = SupermaskLinear(in_features=in1, out_features=hidden_size, sparsity=args.sparsity, bias=args.bias)
        self.fc2 = SupermaskLinear(in_features=hidden_size, out_features=hidden_size, sparsity=args.sparsity, bias=args.bias)
        self.fc3 = SupermaskLinear(in_features=hidden_size, out_features=num_classes, sparsity=args.sparsity, bias=args.bias)

        self.conv1.weight.requires_grad = False
        self.conv2.weight.requires_grad = False

        if args.load_weights is not None:
            state_dict = torch.load(args.load_weights)
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
        for layer in list(state_dict.keys()):
            if 'fc' in layer:
                state_dict[layer] = getattr(self, layer[:3]).weight
                state_dict[layer[:3]+'.scores'] = getattr(self, layer[:3]).scores

        self.load_state_dict(state_dict)


