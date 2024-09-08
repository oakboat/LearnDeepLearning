import torch

x = torch.arange(1, 5, dtype=torch.float, requires_grad=True)
with torch.no_grad():
    x += 2
print(x)
y = x**2
y.backward(torch.ones_like(y))
print(x.grad)