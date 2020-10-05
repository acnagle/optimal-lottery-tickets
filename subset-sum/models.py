import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoLayerFC(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, bias=None):
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
    def __init__(self, input_size, hidden_size, num_classes, bias=None):
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

