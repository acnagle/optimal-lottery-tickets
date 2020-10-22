import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.red_fc import RedLayer

class RedTwoLayerFC(nn.Module):
    def __init__(self, input_size, num_classes, args):
        super(RedTwoLayerFC, self).__init__()
        self.red1 = RedLayer(in_features=input_size, out_features=args.hidden_size, sparsity=args.sparsity, redundancy=args.redundancy, bias=args.bias, use_relu=args.use_relu)
        self.red2 = RedLayer(in_features=args.hidden_size, out_features=num_classes, sparsity=args.sparsity, redundancy=args.redundancy, bias=args.bias, use_relu=args.use_relu)

    def forward(self, x):
        x = self.red1(x)
        x = F.relu(x)
        x = self.red2(x)

        output = F.log_softmax(x, dim=1)

        return output

class RedFourLayerFC(nn.Module):
    def __init__(self, input_size, num_classes, args):
        super(RedFourLayerFC, self).__init__()
        self.red1 = RedLayer(input_size, args.hidden_size, args.sparsity, args.redundancy, args.bias, args.use_relu)
        self.red2 = RedLayer(args.hidden_size, args.hidden_size, args.sparsity, args.redundancy, args.bias, args.use_relu)
        self.red3 = RedLayer(args.hidden_size, args.hidden_size, args.sparsity, args.redundancy, args.bias, args.use_relu)
        self.red4 = RedLayer(args.hidden_size, num_classes, args.sparsity, args.redundancy, args.bias, args.use_relu)

    def forward(self, x):
        x = self.red1(x)
        x = F.relu(x)
        x = self.red2(x)
        x = F.relu(x)
        x = self.red3(x)
        x = F.relu(x)
        x = self.red4(x)

        output = F.log_softmax(x, dim=1)

        return output

class RedLeNet5(nn.Module):
    def __init__(self, input_size, num_classes, args):
        super(RedLeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5), padding=2, bias=args.bias)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5), bias=args.bias)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.red1 = RedLayer(400, 120, args.sparsity, args.redundancy, args.bias, use_relu=args.use_relu)
        self.red2 = RedLayer(120, 84, args.sparsity, args.redundancy, args.bias, use_relu=args.use_relu)
        self.red3 = RedLayer(84, num_classes, args.sparsity, args.redundancy, args.bias, use_relu=args.use_relu)

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

        x = self.red1(x)
        x = F.relu(x)

        x = self.red2(x)
        x = F.relu(x)

        x = self.red3(x)

        output = F.log_softmax(x, dim=-1)

        return output

    def __load_weights(self, state_dict):
        for layer in list(state_dict.keys()):
            if 'fc' in layer:
                fc_layer_num = layer[2]
                state_dict.pop(layer)
                state_dict['red'+fc_layer_num+'.weight1'] = getattr(self, 'red'+fc_layer_num).weight1
                state_dict['red'+fc_layer_num+'.scores1'] = getattr(self, 'red'+fc_layer_num).scores1
                state_dict['red'+fc_layer_num+'.weight2'] = getattr(self, 'red'+fc_layer_num).weight2
                state_dict['red'+fc_layer_num+'.scores2'] = getattr(self, 'red'+fc_layer_num).scores2

        self.load_state_dict(state_dict)


