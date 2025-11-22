import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=(128, 64), activation=nn.ReLU):
        super().__init__()
        layers = []
        in_features = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_features, h))
            layers.append(activation())
            in_features = h
        layers.append(nn.Linear(in_features, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Accept either (batch, features) or (batch, seq_len, features)
        if x.dim() == 3:
            x = x.reshape(x.size(0), -1)
        return self.net(x)


def _make_optimizer(model, optimizer, lr):
    opt = optimizer.upper()
    if opt == "ADAM":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if opt == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr)
    if opt == "RMS":
        return torch.optim.RMSprop(model.parameters(), lr=lr)
    # fallback
    return torch.optim.Adam(model.parameters(), lr=lr)


def _make_criterion(criterion_str):
    c = criterion_str.upper()
    if c == "MSE":
        return nn.MSELoss()
    if c == "MAE":
        return nn.L1Loss()
    if c == "BCE":
        return nn.BCEWithLogitsLoss()
    return nn.MSELoss()


def build(input_size, output_size, optimizer='ADAM', criterion='MSE', lr=0.001):
    """Return (model, optimizer_instance, criterion_instance) for an MLP."""
    model = MLP(input_size=input_size, output_size=output_size)
    # If a PyTorch optimizer instance was provided, use it directly.
    if isinstance(optimizer, torch.optim.Optimizer):
        opt = optimizer
        print("Using provided optimizer instance.")
    # If a PyTorch optimizer class was provided, instantiate it with model parameters.
    elif isinstance(optimizer, type) and issubclass(optimizer, torch.optim.Optimizer):
        opt = optimizer(model.parameters(), lr=lr)
        print("Using provided optimizer class.")
    else:
        opt = _make_optimizer(model, optimizer, lr)
    crit = _make_criterion(criterion)
    return model, opt, crit
