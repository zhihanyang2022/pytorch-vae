import os
import torch
import torch.nn as nn
import torch.distributions as d
import torch.optim as optim


class Decoder(nn.Module):
    """Take in latent vectors, output distribution over images"""

    def __init__(self, z_dim, x_dim):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.mus = nn.Sequential(
            nn.Linear(z_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            # nn.Dropout(0.9),
            nn.Linear(500, x_dim),
            nn.Sigmoid()  # -> (0, 1)
        )

    def forward(self, zs):
        return d.Independent(d.ContinuousBernoulli(self.mus(zs)), 1)


class Encoder(nn.Module):
    """Take in image, output distribution over latent vectors"""

    def __init__(self, z_dim, x_dim):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.shared = nn.Sequential(
            nn.Linear(x_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            # nn.Dropout(0.9)
        )
        self.mus = nn.Linear(500, z_dim)
        self.sigmas = nn.Sequential(
            nn.Linear(500, z_dim),
            nn.Softplus()  # -> (0, infinity)
        )

    def forward(self, xs):
        xs = self.shared(xs)
        return d.Independent(d.Normal(self.mus(xs), self.sigmas(xs)), 1)


class ImageDLVM:

    def __init__(self, z_dim, x_dim, lr=1e-3):

        # hyper-parameters
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.lr = lr

        # describes the generative process
        self.p_z = d.Independent(d.Normal(torch.zeros(z_dim), torch.ones(z_dim)), 1)
        self.p_x_given_z = Decoder(z_dim=z_dim, x_dim=x_dim)

        # required for approximate posterior inference
        self.q_z_given_x = Encoder(z_dim=z_dim, x_dim=x_dim)

        # gradient-based optimizers
        self.p_x_given_z_opt = optim.Adam(self.p_x_given_z.parameters(), lr=lr)
        self.q_z_given_x_opt = optim.Adam(self.q_z_given_x.parameters(), lr=lr)

    def fit(self, xs):
        self.p_x_given_z.train()
        self.q_z_given_x.train()
        posterior_over_zs = self.q_z_given_x(xs)
        # kl-divergence
        kl = d.kl.kl_divergence(posterior_over_zs, self.p_z).mean()
        # reconstruction
        zs = posterior_over_zs.rsample()  # sample using the re-parametrization trick
        rec = self.p_x_given_z(zs).log_prob(xs).mean()
        # elbo
        elbo = - kl + rec
        # backpropagation and gradient step
        loss = - elbo
        self.p_x_given_z_opt.zero_grad()
        self.q_z_given_x_opt.zero_grad()
        loss.backward()
        self.p_x_given_z_opt.step()
        self.q_z_given_x_opt.step()
        return {
            "kl": float(kl),  # this is at least 0; the smaller the better
            "rec": float(rec),  # the larger the better
            "elbo": float(elbo)  # the larger the better
        }

    def encode(self, xs):
        with torch.no_grad():
            return self.q_z_given_x(xs).mean

    def sample(self, n):
        self.p_x_given_z.eval()
        self.q_z_given_x.eval()
        with torch.no_grad():
            return self.p_x_given_z(self.p_z.sample((n, ))).mean

    def save(self, save_dir):
        torch.save(self.p_x_given_z.state_dict(), os.path.join(save_dir, "p_x_given_z.pth"))
        torch.save(self.q_z_given_x.state_dict(), os.path.join(save_dir, "q_z_given_x.pth"))

    def load(self, save_dir):
        self.p_x_given_z.load_state_dict(
            torch.load(os.path.join(save_dir, "p_x_given_z.pth"), map_location=torch.device("cpu"))
        )
        self.q_z_given_x.load_state_dict(
            torch.load(os.path.join(save_dir, "q_z_given_x.pth"), map_location=torch.device("cpu"))
        )
