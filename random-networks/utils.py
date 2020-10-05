import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

# this method is modified from https://github.com/allenai/hidden-networks/blob/master/simple_mnist_example.py. Credit goes to Ramanujan, et al.
class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # get the supermask by sorting the scores and using the top k
        out = scores.clone()

        flat_out = out.flatten()
        _, idx = flat_out.sort()
        j = int(k * scores.numel())

        # flat_out and out access the same memory.
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None

# this method is modified from https://github.com/allenai/hidden-networks/blob/master/simple_mnist_example.py. Credit goes to Ramanujan, et al.
class SupermaskLinear(nn.Linear):
    def __init__(self, sparsity, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sparsity = sparsity

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # initialize the weights
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # turn the gradient on the weights off
        self.weight.requires_grad = False

    def forward(self, x):
        subnet = GetSubnet.apply(self.scores.abs(), self.sparsity)
        w = self.weight * subnet

        return F.linear(x, w, self.bias)

def train(model, device, train_loader, optimizer, criterion, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx+1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, criterion, test_loader, batch_size):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= np.ceil(len(test_loader.dataset) / batch_size)
    test_acc = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        test_acc))

    return test_acc



