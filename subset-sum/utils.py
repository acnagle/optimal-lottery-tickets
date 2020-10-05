import torch
import numpy as np

import sys

import gurobipy as gp
from gurobipy import GRB

def FindSubset(w, a, eps, n, output_flag=False, check_w_lt_eps=False):
    subset_sum = None
    num_used = 0    # number of a_i terms used in the subset sum
    
    if check_w_lt_eps and (abs(w) <= eps):    # check if the magnitude of w is less than eps
        subset_sum = 0
    else:
        m = gp.Model('mip1')
        m.Params.OutputFlag = output_flag

        x = m.addVars(n, vtype=GRB.BINARY)
        z = m.addVar(vtype=GRB.CONTINUOUS)
        m.setObjective(z, GRB.MINIMIZE)
        m.addConstr(w - x.prod(a) <= z)
        m.addConstr(-w + x.prod(a) <= z)
        m.addConstr(w - x.prod(a) <= eps)
        m.addConstr(-w + x.prod(a) <= eps)
        m.Params.MIPGap = 0.01
        m.optimize()

        if m.status == 2:   # feasible solution found
            subset = []
            for i in range(len(x)):
                if round(x[i].x) > 0:
                    subset.append(a[i])
            subset_sum = sum(subset)
            num_used = len(subset)

            if output_flag:     # print verbose information
                diff = abs(subset_sum - w)
                print('\n' + '-' * 96) 
                print('\nNumber of elements in subset:', num_used)
                print('\nValues used to approximate w:', subset)
                print('\nSubset sum:', subset_sum, 'is approximately equal to', w)
                print('\nDifference between subset sum and w:', diff, ', epsilon =', eps)
                print('This difference is less than epsilon:', diff <= eps)
        else:
            print('\nFeasible solution not found for weight value', w, 'and coefficients', a)
            print('Try increasing c, decreasing epsilon, or both.')
            sys.exit(0)

    return subset_sum, num_used

def train(model, device, train_loader, optimizer, criterion, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.reshape(-1, 28*28).to(device), target.to(device)
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
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= np.ceil(len(test_loader.dataset) / batch_size)
    test_acc = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        test_acc))

    return test_acc



