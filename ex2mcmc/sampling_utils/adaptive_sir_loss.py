import torch
from torch import nn
import torch.nn.functional as F


def get_optimizer(parameters, optimizer="Adam", lr=1e-3, weight_decay=1e-5):
    if optimizer == "Adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError


def get_loss(loss):
    if loss == "mix_kl":
        return MixKLLoss
    if loss == "forward_kl":
        return forward_kl
    if loss == "backward_kl":
        return backward_kl
    else:
        raise NotImplementedError


def forward_kl(target, proposal, flow, x):
    x_ = x.detach().requires_grad_()
    z, log_jac = flow.forward(x_)
    est = target.log_prob(x_) - (proposal.log_prob(z) + log_jac)
    grad_est = -proposal.log_prob(z) - log_jac
    return est.mean(), grad_est.mean()


def backward_kl(target, proposal, flow, x):
    z = proposal.sample(x.shape[:-1])
    x, log_jac_inv = flow.inverse(z)
    est = proposal.log_prob(z) - log_jac_inv - target.log_prob(x)
    grad_est = -log_jac_inv - target.log_prob(x)
    return est.mean(), grad_est.mean()


def mix_kl(
    target,
    proposal,
    flow,
    y,
    alpha=0.99,
    gamma=None,
):
    est_f, grad_est_f = forward_kl(target, proposal, flow, y)
    est_b, grad_est_b = backward_kl(target, proposal, flow, y)

    if torch.isnan(grad_est_b).item():
        grad_est_b = 0

    return (
        alpha * est_f + (1.0 - alpha) * est_b,
        alpha * grad_est_f + (1.0 - alpha) * grad_est_b,
    )


class MixKLLoss(nn.Module):
    def __init__(
        self,
        target,
        proposal,
        flow,
        alpha=0.99,
        beta=0.0,
        gamma=0.99,
    ):  # .2):#.99):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.flow = flow
        self.target = target
        self.proposal = proposal

    def forward(self, y, acc_rate=1.0, alpha=None, beta=None, gamma=None):
        alpha = alpha if alpha is not None else self.alpha
        beta = beta if beta is not None else self.beta
        gamma = gamma if gamma is not None else self.gamma

        return mix_kl(
            self.target,
            self.proposal,
            self.flow,
            y,
            alpha=alpha,
        )

class Vae_loss(nn.Module):
    def __init__(
        self,
        flow,
        kld_weight,
        beta=0.0,
    ):  # .2):#.99):
        super().__init__()

        # self.alpha = alpha
        self.beta = beta
        # self.gamma = gamma
        self.flow = flow
        # self.target = target
        # self.proposal = proposal
        self.kld_weight = kld_weight

    def forward(self, recons, input, mu, log_var):
        # alpha = alpha if alpha is not None else self.alpha
        # beta = beta if beta is not None else self.beta
        # gamma = gamma if gamma is not None else self.gamma

        recons_loss =F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = recons_loss + self.beta * self.kld_weight * kld_loss
        return loss
        
        # recons = args[0]
        # input = args[1]
        # mu = args[2]
        # log_var = args[3]
        # kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        # recons_loss =F.mse_loss(recons, input)

        # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        # if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
        #     loss = recons_loss + self.beta * kld_weight * kld_loss
        # elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
        #     self.C_max = self.C_max.to(input.device)
        #     C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
        #     loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
        # else:
        #     raise ValueError('Undefined loss type.')

        # return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}