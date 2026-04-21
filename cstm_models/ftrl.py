import torch
from torch.optim import Optimizer


class FTRL(Optimizer):
    """
    FTRL-Proximal optimizer (Follow The Regularized Leader).
    From: McMahan et al., "Ad Click Prediction: a View from the Trenches" (Google, 2013).

    Designed for online learning — per-coordinate adaptive learning rates with
    L1 (sparsity) and L2 regularization built in. Better theoretical regret
    bounds than SGD or Adam for non-stationary online settings.

    Per-coordinate update:
        n_t  = n_{t-1} + g_t²
        σ    = (√n_t - √n_{t-1}) / alpha
        z_t  = z_{t-1} + g_t - σ * w_{t-1}

        if |z_t| <= lambda1:   w_t = 0
        else:
            w_t = -(z_t - lambda1 * sign(z_t)) / (lambda2 + (beta + √n_t) / alpha)

    Args:
        params:   model parameters
        alpha:    initial learning rate (default 0.01)
        beta:     smoothing term — controls early-step LR (default 1.0)
        lambda1:  L1 penalty — promotes sparsity (default 0.0)
        lambda2:  L2 penalty — stabilises weights (default 0.0)
    """

    def __init__(self, params, alpha=0.01, beta=1.0, lambda1=0.0, lambda2=0.0):
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        if beta < 0:
            raise ValueError(f"beta must be >= 0, got {beta}")
        defaults = dict(alpha=alpha, beta=beta, lambda1=lambda1, lambda2=lambda2)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            alpha   = group["alpha"]
            beta    = group["beta"]
            lambda1 = group["lambda1"]
            lambda2 = group["lambda2"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["n"] = torch.zeros_like(p)
                    state["z"] = torch.zeros_like(p)

                n, z = state["n"], state["z"]

                # n_t = n_{t-1} + g²
                n_new = n + g * g

                # σ = (√n_t - √n_{t-1}) / alpha
                sigma = (n_new.sqrt() - n.sqrt()) / alpha

                # z_t = z_{t-1} + g - σ * w
                z.add_(g - sigma * p)
                n.copy_(n_new)

                # w_t: apply proximal L1 threshold
                if lambda1 > 0:
                    sign_z = z.sign()
                    z_thresh = z.abs() - lambda1
                    mask = z_thresh > 0
                    denom = lambda2 + (beta + n.sqrt()) / alpha
                    p.copy_(torch.where(mask, -sign_z * z_thresh / denom, torch.zeros_like(p)))
                else:
                    denom = lambda2 + (beta + n.sqrt()) / alpha
                    p.copy_(-z / denom)

        return loss
