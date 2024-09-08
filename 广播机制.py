import torch

x = torch.arange(3).reshape([1,1,3])
print(x)
y = torch.arange(3).reshape([1,3,1])
print(y)
z = torch.arange(3).reshape([3,1,1])
print(z)
print(x+y+z)