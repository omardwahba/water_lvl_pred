import torch
import torch.nn as nn


class CNNForecast(nn.Module):
    def __init__(self, input_size, output_size, channels=(32, 64), kernel_size=3):
        super().__init__()
        # Expect input shape (batch, seq_len, features)
        # Conv1d expects (batch, channels=features, seq_len)
        layers = []
        in_channels = input_size
        for ch in channels:
            layers.append(nn.Conv1d(in_channels, ch, kernel_size=kernel_size, padding=kernel_size//2))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(ch))
            in_channels = ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_channels, output_size)

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # -> (batch, features, seq_len)
        out = self.conv(x)
        out = self.pool(out).squeeze(-1)
        return self.fc(out)


def _make_optimizer(model, optimizer, lr):
    o = optimizer.upper()
    if o == 'ADAM':
        return torch.optim.Adam(model.parameters(), lr=lr)
    if o == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr)
    return torch.optim.Adam(model.parameters(), lr=lr)


def _make_criterion(criterion_str):
    c = criterion_str.upper()
    if c == 'MSE':
        return nn.MSELoss()
    if c == 'MAE':
        return nn.L1Loss()
    if c == 'BCE':
        return nn.BCEWithLogitsLoss()
    return nn.MSELoss()


def build(input_size, output_size, optimizer='ADAM', criterion='MSE', lr=0.001):
    model = CNNForecast(input_size=input_size, output_size=output_size)
    opt = _make_optimizer(model, optimizer, lr)
    crit = _make_criterion(criterion)
    return model, opt, crit
