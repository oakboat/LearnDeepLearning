import torch

def f(x):
    return 4*x**2-6*x+8

x = torch.tensor([50.], requires_grad=True)
for _ in range (200):
    y = f(x)
    print(x, y)
    y.backward()
    with torch.no_grad():
        x -= x.grad * 0.01
        x.grad.zero_()
        print(x)