import torch
import torch.nn as nn


class LSTMForecast(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, (hn, cn) = self.lstm(x)
        # take last time-step output
        last = out[:, -1, :]
        return self.fc(last)


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
    model = LSTMForecast(input_size=input_size, output_size=output_size)
    opt = _make_optimizer(model, optimizer, lr)
    crit = _make_criterion(criterion)
    return model, opt, crit
