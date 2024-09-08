import torch
import matplotlib.pyplot as plt

x = torch.arange(-2, 2, 0.01, dtype=torch.float64, requires_grad=True)
y1 = torch.sin(x)
y2 = y1.sum()
y2.backward()
plt.plot(x.detach().numpy(), y1.detach().numpy())
plt.plot(x.detach().numpy(), x.grad.detach().numpy())
plt.show()