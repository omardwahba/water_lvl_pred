import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        if x.dim() == 3:
            x = x.reshape(x.size(0), -1)
        return self.linear(x)


def _make_optimizer(model, optimizer, lr):
    o = optimizer.upper()
    if o == 'ADAM':
        return torch.optim.Adam(model.parameters(), lr=lr)
    if o == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr)
    return torch.optim.Adam(model.parameters(), lr=lr)


def _make_criterion(criterion_str):
    c = criterion_str.upper()
    if c == 'BCE':
        return nn.BCEWithLogitsLoss()
    if c == 'MSE':
        return nn.MSELoss()
    if c == 'MAE':
        return nn.L1Loss()
    return nn.BCEWithLogitsLoss()


def build(input_size, output_size, optimizer='ADAM', criterion='MSE', lr=0.001):
    model = LogisticRegression(input_size=input_size, output_size=output_size)
    opt = _make_optimizer(model, optimizer, lr)
    crit = _make_criterion(criterion)
    return model, opt, crit
