"""Single-omics AE + GCNII graph autoencoder."""

from __future__ import annotations

import torch
from torch import nn

from model.AE import AE
from model.GCNIIIGAE import GCNIIStyleIGAE
from model.SimpleIGAE import SimpleIGAE


class SingleOmicsSpaNicheGraph(nn.Module):
    """One RNA branch with attribute-anchored graph propagation."""

    def __init__(
        self,
        n_input: int,
        n_z: int = 40,
        alpha: float = 0.3,
        lamda: float = 0.4,
        graph_encoder_variant: str = "gcnii",
    ):
        super().__init__()
        self.n_input = int(n_input)
        self.n_z = int(n_z)
        self.sigma = 1.0
        self.graph_encoder_variant = str(graph_encoder_variant).lower()
        self.ae = AE(128, 256, n_z, n_z, 256, 128, n_input, n_z)
        if self.graph_encoder_variant == "gcnii":
            self.gae = GCNIIStyleIGAE(
                gae_n_enc_1=128,
                gae_n_enc_2=256,
                gae_n_enc_3=n_z,
                gae_n_dec_1=n_z,
                gae_n_dec_2=256,
                gae_n_dec_3=128,
                n_input=n_input,
                ae_n_enc_1=128,
                ae_n_enc_2=256,
                ae_n_enc_3=n_z,
                alpha=alpha,
                lamda=lamda,
            )
        elif self.graph_encoder_variant == "simplegcn":
            self.gae = SimpleIGAE(128, 256, n_z, n_z, 256, 128, n_input)
        else:
            raise ValueError(f"Unsupported graph_encoder_variant: {graph_encoder_variant}")
        self.last_dynamic_graph = None

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        if int(x.shape[1]) != self.n_input:
            raise ValueError(f"RNA input dimension {x.shape[1]} != expected {self.n_input}.")
        z_ae1, z_ae2, z_ae3 = self.ae.encoder(x)
        if self.graph_encoder_variant == "gcnii":
            z_igae3, z_igae_adj = self.gae.encoder(
                x,
                adj,
                ae_anchors=(z_ae1, z_ae2, z_ae3),
            )
        else:
            z_igae3, z_igae_adj = self.gae.encoder(x, adj)
        z_tilde = (1.0 - self.sigma) * z_ae3 + self.sigma * z_igae3
        x_hat = self.ae.decoder(z_ae3)
        z_hat, z_hat_adj = self.gae.decoder(z_igae3, adj)
        return x_hat, z_hat, z_igae_adj + z_hat_adj, z_ae3, z_igae3, z_tilde
