import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.red_fc import RedLayer

class RedTwoLayerFC(nn.Module):
    def __init__(self, input_size, num_classes, args):      #TODO: Why isn't all the RedLayer() info showing up when I print out the model? MIGHT BE BECUASE I NEED TO INCLUDE .__INIT__(*ARGS, **KWARGS) BELOW
        super(RedTwoLayerFC, self).__init__()
        self.red1 = RedLayer(in_features=input_size, out_features=args.hidden_size, sparsity=args.sparsity, redundancy=args.r, bias=args.bias, use_relu=args.use_relu)
        self.red2 = RedLayer(in_features=args.hidden_size, out_features=num_classes, sparsity=args.sparsity, redundancy=args.r, bias=args.bias, use_relu=args.use_relu)

    def forward(self, x):
        x = self.red1(x)
        x = F.relu(x)
        x = self.red2(x)

        output = F.log_softmax(x, dim=1)

        return output

class RedFourLayerFC(nn.Module):
    def __init__(self, input_size, num_classes, args):
        super(RedFourLayerFC, self).__init__()
        self.red1 = RedLayer(input_size, args.hidden_size, args.sparsity, args.r, args.bias, args.use_relu)
        self.red2 = RedLayer(args.hidden_size, args.hidden_size, args.sparsity, args.r, args.bias, args.use_relu)
        self.red3 = RedLayer(args.hidden_size, args.hidden_size, args.sparsity, args.r, args.bias, args.use_relu)
        self.red4 = RedLayer(args.hidden_size, num_classes, args.sparsity, args.r, args.bias, args.use_relu)

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

class RedFCLeNet5(nn.Module):       # TODO: rename to RedLeNet5FC (copy the name convention for fully connected networks)
    def __init__(self, input_size, num_classes, args):
        super(RedFCLeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5), padding=2, bias=args.bias)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5), bias=args.bias)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.red1 = RedLayer(400, 120, args.sparsity, args.redundancy, args.bias, use_relu=args.use_relu)
        self.red2 = RedLayer(120, 84, args.sparsity, args.redundancy, args.bias, use_relu=args.use_relu)
        self.red3 = RedLayer(84, num_classes, args.sparsity, args.redundancy, args.bias, use_relu=args.use_relu)

#        if state_dict is not None:
#            # freeze weights
#            self.conv1.weight.requires_grad = False     # TODO: freeze and load weights in main
#            self.conv2.weight.requires_grad = False
#
#            self.__load_weights(state_dict)

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
        # weights that are loaded in are assumed to be coming from the LeNet5 architecture
        
        # remove fully connected layer weights
        state_dict.popitem()
        state_dict.popitem()
        state_dict.popitem()

        # add redundant weights and their respective scores
        state_dict['red1.weight1'] = self.red1.weight1
        state_dic['red1.scores1'] = self.red1.scores1
        state_dict['red1.weight2'] = self.red1.weight2
        state_dict['red1.scores2'] = self.red1.scores2
        state_dict['red2.weight1'] = self.red2.weight1
        state_dict['red2.scores1'] = self.red2.scores1
        state_dict['red2.weight2'] = self.red2.weight2
        state_dict['red2.scores2'] = self.red2.scores2
        state_dict['red3.weight1'] = self.red3.weight1
        state_dict['red3.scores1'] = self.red3.scores1
        state_dict['red3.weight2'] = self.red3.weight2
        state_dict['red3.scores2'] = self.red3.scores2

        self.load_state_dict(state_dict)

vgg_cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class RedFCVGG(nn.Module):
    def __init__(self, vgg_name, num_classes, sparsity, redundancy, bias, use_relu, state_dict=None):
        super(VGG, self).__init__()
        self.features = self._make_layers(vgg_cfg[vgg_name])
        self.classifier = nn.Sequential(
            RedLayer(512, 512, sparsity, redundancy, bias, use_relu),    # VGG typically uses 4096 nodes per FC later, but with our architecture, this would take too long to train
            nn.ReLU(True),
            nn.Dropout(),
            RedLayer(512, 512, sparsity, redundancy, bias, use_relu),
            nn.ReLU(True),
            nn.Dropout(),
            RedLayer(512, num_classes, sparsity, redundancy, bias, use_relu)
        )
        
        if state_dict is not None:
            # freeze weights
            for idx, layer in enumerate(self.features.children()):
                if isinstance(layer, nn.Conv2d):
                    self.features[idx].weight.requires_grad = False

            self.__load_weights(state_dict)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return output

    def _make_layers(self, vgg_cfg):
        layers = []
        in_channels = 3
        for x in vgg_cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def __load_weights(self, state_dict):
        # weights that are loaded in are assumed to be coming from the VGG architecture
        
        # remove fully connected layer weights
        state_dict.popitem()
        state_dict.popitem()
        state_dict.popitem()

        # add redundant weights and their respective scores
        red_layer_idx = 1
        for idx, layer in enumerate(self.classifier.children()):
            if isinstance(layer, RedLayer):
                state_dict['red'+str(red_layer_idx)+'.weight1'] = self.classifier[idx].weight1
                state_dict['red'+str(red_layer_idx)+'.scores1'] = self.classifier[idx].scores1
                state_dict['red'+str(red_layer_idx)+'.weight2'] = self.classifier[idx].weight2
                state_dict['red'+str(red_layer_idx)+'.scores2'] = self.classifier[idx].scores1
                red_layer_idx += 1

#        state_dict['red1.weight1'] = self.red1.weight1
#        state_dict['red1.scores1'] = self.red1.scores1
#        state_dict['red1.weight2'] = self.red1.weight2
#        state_dict['red1.scores2'] = self.red1.scores2
#        state_dict['red2.weight1'] = self.red2.weight1
#        state_dict['red2.scores1'] = self.red2.scores1
#        state_dict['red2.weight2'] = self.red2.weight2
#        state_dict['red2.scores2'] = self.red2.scores2
#        state_dict['red3.weight1'] = self.red3.weight1
#        state_dict['red3.scores1'] = self.red3.scores1
#        state_dict['red3.weight2'] = self.red3.weight2
#        state_dict['red3.scores2'] = self.red3.scores2

        self.load_state_dict(state_dict)


