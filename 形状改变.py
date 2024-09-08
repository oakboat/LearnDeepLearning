import torch

x = torch.arange(-10, 10, dtype=torch.float)
x = x.reshape(2, -1)
x.requires_grad = True
y = torch.sum(x, dim=1)
y.backward(torch.ones(2)*0)
print(x.grad)