import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoLayerFC(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, bias):
        super(TwoLayerFC, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc2 = nn.Linear(hidden_size, num_classes, bias=bias)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)

        return output

class FourLayerFC(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, bias):
        super(FourLayerFC, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.fc3 = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.fc4 = nn.Linear(hidden_size, num_classes, bias=bias)

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
    def __init__(self, num_classes, bias):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5), padding=2, bias=bias)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5), bias=bias)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.fc1 = nn.Linear(400, 120, bias=bias)
        self.fc2 = nn.Linear(120, 84, bias=bias)
        self.fc3 = nn.Linear(84, num_classes, bias=bias)

        self.conv1.weight.requires_grad = False
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


