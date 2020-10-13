import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoLayerFC(nn.Module):
    def __init__(self, input_size, num_classes, args):
        super(TwoLayerFC, self).__init__()
        self.fc1 = nn.Linear(input_size, args.hidden_size, bias=args.bias)
        self.fc2 = nn.Linear(args.hidden_size, num_classes, bias=args.bias)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)

        return output

class FourLayerFC(nn.Module):
    def __init__(self, input_size, num_classes, args):
        super(FourLayerFC, self).__init__()
        self.fc1 = nn.Linear(input_size, args.hidden_size, bias=args.bias)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size, bias=args.bias)
        self.fc3 = nn.Linear(args.hidden_size, args.hidden_size, bias=args.bias)
        self.fc4 = nn.Linear(args.hidden_size, num_classes, bias=args.bias)

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

class LeNet5(nn.Module):
    def __init__(self, input_size, num_classes, args):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5), padding=2, bias=args.bias)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5), bias=args.bias)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.fc1 = nn.Linear(400, 120, bias=args.bias)
        self.fc2 = nn.Linear(120, 84, bias=args.bias)
        self.fc3 = nn.Linear(84, num_classes, bias=args.bias)

        self.conv1.weight.requires_grad = False     # TODO: This is assuming a network is loaded in... get rid of this an add flexibility
        self.conv2.weight.requires_grad = False

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

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
