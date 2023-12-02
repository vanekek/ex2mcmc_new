import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
from abc import abstractmethod

class BaseVAE(nn.Module):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


class BetaVAE(BaseVAE):

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 beta: int = 4,
                 gamma:float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type:str = 'B',
                 **kwargs) -> None:
        super(BetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

class VaeMCMC:
    def __init__(self, target, proposal, device, flow, mcmc_call: callable, **kwargs):
        self.flow = flow
        self.proposal = proposal
        self.target = target
        self.device = device
        self.batch_size = kwargs.get("batch_size", 64)
        self.mcmc_call = mcmc_call
        self.grad_clip = kwargs.get("grad_clip", 1.0)
        self.jump_tol = kwargs.get("jump_tol", 1e6)
        optimizer = kwargs.get("optimizer", "adam")
        loss = kwargs.get("loss", "mix_kl")
        self.flow.to(self.device)
        if isinstance(loss, (Callable, nn.Module)):
            self.loss = loss
        elif isinstance(loss, str):
            self.loss = get_loss(loss)(self.target, self.proposal, self.flow)
        else:
            ValueError

        lr = kwargs.get("lr", 1e-3)
        wd = kwargs.get("wd", 1e-4)
        if isinstance(optimizer, torch.optim.Optimizer):
            self.optimizer = optimizer
        elif isinstance(optimizer, str):
            if optimizer.lower() == "adam":
                self.optimizer = torch.optim.Adam(
                    flow.parameters(), lr=lr, weight_decay=wd
                )

        self.loss_hist = []

    def train_step(self, inp=None, alpha=0.5, do_step=True, inv=True):
        if do_step:
            self.optimizer.zero_grad()
        if inp is None:
            inp = self.proposal.sample((self.batch_size,))
        elif inv:
            inp, _ = self.flow.forward(inp)
        out = self.mcmc_call(inp, self.target, self.proposal, flow=self.flow)
        if isinstance(out, Tuple):
            acc_rate = out[1].mean()
            out = out[0]
        else:
            acc_rate = 1
        out = out[-1]
        out = out.to(self.device)
        nll = -self.target.log_prob(out).mean().item()

        if do_step:
            loss_est, loss = self.loss(out, acc_rate=acc_rate, alpha=alpha)

            if (
                len(self.loss_hist) > 0
                and loss.item() - self.loss_hist[-1] > self.jump_tol
            ) or torch.isnan(loss):
                print("KL wants to jump, terminating learning")
                return out, nll

            self.loss_hist = self.loss_hist[-500:] + [loss_est.item()]
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.flow.parameters(),
                self.grad_clip,
            )
            self.optimizer.step()

        return out, nll

    def train(self, n_steps=100, start_optim=10, init_points=None, alpha=None):
        samples = []
        inp = self.proposal.sample((self.batch_size,))

        neg_log_likelihood = []

        for step_id in trange(n_steps):
            # if alpha is not None:
            #    if isinstance(alpha, Callable):
            #        a = alpha(step_id)
            #    elif isinstance(alpha, float):
            #        a = alpha
            # else:
            a = min(0.5, 3 * step_id / n_steps)

            out, nll = self.train_step(
                alpha=a,
                do_step=step_id >= start_optim,
                inp=init_points if step_id == 0 and init_points is not None else inp,
                inv=True,
            )
            inp = out.detach().requires_grad_()
            samples.append(inp.detach().cpu())

            neg_log_likelihood.append(nll)

        return samples, neg_log_likelihood

    def sample(self):
        pass