"""Autoencoder Variacional (VAE) para Fashion-MNIST.

Arquitetura MLP totalmente conectada. O encoder produz media e log-variancia
do posterior aproximado q(z|x); o decoder reconstroi a imagem a partir de z.
A funcao de custo e o ELBO negativo: BCE de reconstrucao somada a divergencia
KL com peso beta.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_DIM = 784


def _mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class VAE(nn.Module):
    def __init__(self, latent_dim: int, hidden: tuple[int, ...] = (512, 256)):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden = tuple(hidden)

        enc_sizes = [INPUT_DIM, *hidden]
        self.encoder = _mlp(enc_sizes)
        self.fc_mu = nn.Linear(hidden[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden[-1], latent_dim)

        dec_sizes = [latent_dim, *reversed(hidden), INPUT_DIM]
        self.decoder = _mlp(dec_sizes)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decoder(z))

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def elbo_loss(recon: torch.Tensor, x: torch.Tensor,
              mu: torch.Tensor, logvar: torch.Tensor,
              beta: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Retorna (loss total, termo de reconstrucao, termo KL), por amostra."""
    batch = x.size(0)
    recon_term = F.binary_cross_entropy(recon, x, reduction="sum") / batch
    kl_term = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch
    loss = recon_term + beta * kl_term
    return loss, recon_term, kl_term
