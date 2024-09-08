import time
import torch
from torch import nn
from d2l import torch as d2l
from torch.utils.tensorboard import SummaryWriter

def evaluate_accuracy_gpu(net, data_iter, device):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr = lr)
    loss = nn.CrossEntropyLoss()
    timer, num_batchs = d2l.Timer(), len(train_iter)
    log_dir = "runs/" + net.__class__.__name__ + time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    writer = SummaryWriter(log_dir)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            writer.add_scalar('training loss',
                    train_l,
                    epoch * num_batchs + i)
            writer.add_scalar('training accuracy',
                    train_acc,
                    epoch * num_batchs + i)
        test_acc = evaluate_accuracy_gpu(net, test_iter, device)
        writer.add_scalar('test accuracy',
                        test_acc,
                        epoch * num_batchs + i)
        print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
