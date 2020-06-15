## Test on dummy data
## TODO: Use this on tf implementation and check values
# wrap your optimizer
import sys
sys.path.append('../')

import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
import random

from operator import mul
from functools import reduce

from model import Net
from pcgrad import pc_grad_update

net = Net()

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
inp = torch.randn(3, 3)
mse_loss = nn.MSELoss()

output = net(inp)
output = output.view(1, -1)

target = torch.randn(3, 3)  # a dummy target, for example
target = target.view(1, -1)

loss = mse_loss(output, target)
loss.backward()

grads = []
for p in net.parameters():
  # Simulate 5 different tasks
  grads.append(p.grad)
  
grad_list = [[torch.rand(size=grad.shape) for grad in grads] for i in range(5)]

print(pc_grad_update(grad_list))
print(grad_list)

optimizer.step() 

print(loss)
