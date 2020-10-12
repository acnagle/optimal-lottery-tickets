import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class RedLayer(nn.Module):
    def __init__(self, in_features, out_features, sparsity, redundancy, bias, use_relu):
        super(RedLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity
        self.redundancy = redundancy
        self.bias = bias
        self.use_relu = use_relu

        self.weight1 = nn.Parameter(torch.Tensor(redundancy, in_features))    # weights for the redundant layer
        self.scores1 = nn.Parameter(torch.Tensor(self.weight1.size()))
        self.weight2 = nn.Parameter(torch.Tensor(out_features, in_features * redundancy))
        self.scores2 = nn.Parameter(torch.Tensor(self.weight2.size()))

        std1 = math.sqrt(2. / 1)
        std2 = math.sqrt(2. / in_features)

        # initialize the scores
        nn.init.uniform_(self.scores1, a=-std1, b=std1)    # kaiming uniform
        nn.init.uniform_(self.scores2, a=-std2, b=std2)

        nn.init.normal_(self.weight1, std=std1)   # kaiming normal
        nn.init.normal_(self.weight2, std=std2)
    
        # turn the gradient on the weights off
        self.weight1.requires_grad = False
        self.weight2.requires_grad = False

    def forward(self, x): 
        x = x.view(x.size()[0], 1, -1) 
        x = x.repeat(1, self.redundancy, 1)    # x.size() = torch.Size([B, redundancy, in_features])

        subnet1 = utils.GetSubnet.apply(self.scores1.abs(), self.sparsity)     # get pruning mask
        w1 = self.weight1 * subnet1     # apply pruning mask
    
        x = x * w1    # element-wise multiply input with weights
        
        if self.use_relu:
            x = F.relu(x)

        x = x.permute(0, 2, 1).reshape(x.size()[0], -1)     # x.size() = torch.Size([B, redundancy * in_features])
    
        subnet2 = utils.GetSubnet.apply(self.scores2.abs(), self.sparsity)
        w2 = self.weight2 * subnet2
    
        output = F.linear(x, w2, bias=self.bias)

        return output

