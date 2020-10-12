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


