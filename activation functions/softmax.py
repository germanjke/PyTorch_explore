import torch.nn as nn
import torch

softmax = nn.Softmax(dim=1)
x_input = torch.randn(1,3)
y_ouput = softmax(x_input)
print(x_input)
print(y_ouput)
print(torch.sum(y_ouput, dim=1))