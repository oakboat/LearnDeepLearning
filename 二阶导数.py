import torch

# 示例张量和操作
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2

# 第一次反向传播
y.backward(torch.tensor([1.0, 1.0, 1.0]), retain_graph=True)

# 如果需要再次计算梯度，像这样操作：
y.backward(torch.tensor([1.0, 1.0, 1.0]), retain_graph=True)

print(x.grad)