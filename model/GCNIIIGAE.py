import math

import torch
from torch import nn

from model.SimpleIGAE import SimpleIGAEDecoder


def _graph_mm(adj, features):
    if adj.is_sparse:
        return torch.spmm(adj, features)
    return torch.matmul(adj, features)


class GCNIIStyleLayer(nn.Module):
    """GCNII-style layer with an external attribute anchor."""

    def __init__(
        self,
        in_features,
        out_features,
        ref_features,
        alpha=0.1,
        lamda=0.5,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.lamda = float(lamda)
        self.msg_proj = nn.Linear(in_features, out_features, bias=False)
        self.ref_proj = (
            nn.Identity()
            if int(ref_features) == int(out_features)
            else nn.Linear(ref_features, out_features, bias=False)
        )
        self.out_proj = nn.Linear(out_features, out_features, bias=False)
        self.act = nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.msg_proj.weight)
        if isinstance(self.ref_proj, nn.Linear):
            nn.init.xavier_uniform_(self.ref_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, features, adj, h0, layer_idx, active=True):
        layer_idx = max(int(layer_idx), 1)
        beta = math.log(self.lamda / layer_idx + 1.0)

        message = _graph_mm(adj, self.msg_proj(features))
        residual = self.ref_proj(h0)
        support = (1.0 - self.alpha) * message + self.alpha * residual
        transformed = self.out_proj(support)
        output = (1.0 - beta) * support + beta * transformed

        if active:
            output = self.act(output)
        return output


class GCNIIStyleEncoder(nn.Module):
    def __init__(
        self,
        gae_n_enc_1,
        gae_n_enc_2,
        gae_n_enc_3,
        n_input,
        ae_n_enc_1=None,
        ae_n_enc_2=None,
        ae_n_enc_3=None,
        alpha=0.1,
        lamda=0.5,
    ):
        super().__init__()
        ae_n_enc_1 = gae_n_enc_1 if ae_n_enc_1 is None else ae_n_enc_1
        ae_n_enc_2 = gae_n_enc_2 if ae_n_enc_2 is None else ae_n_enc_2
        ae_n_enc_3 = gae_n_enc_3 if ae_n_enc_3 is None else ae_n_enc_3
        self.gnn_1 = GCNIIStyleLayer(n_input, gae_n_enc_1, ae_n_enc_1, alpha=alpha, lamda=lamda)
        self.gnn_2 = GCNIIStyleLayer(gae_n_enc_1, gae_n_enc_2, ae_n_enc_2, alpha=alpha, lamda=lamda)
        self.gnn_3 = GCNIIStyleLayer(gae_n_enc_2, gae_n_enc_3, ae_n_enc_3, alpha=alpha, lamda=lamda)
        self.fallback_anchor_1 = nn.Linear(n_input, ae_n_enc_1, bias=False)
        self.fallback_anchor_2 = nn.Linear(n_input, ae_n_enc_2, bias=False)
        self.fallback_anchor_3 = nn.Linear(n_input, ae_n_enc_3, bias=False)
        self.s = nn.Sigmoid()
        self.last_boundary_gate_stats = {"enabled": 0}
        self.reset_fallback_parameters()

    def reset_fallback_parameters(self):
        nn.init.xavier_uniform_(self.fallback_anchor_1.weight)
        nn.init.xavier_uniform_(self.fallback_anchor_2.weight)
        nn.init.xavier_uniform_(self.fallback_anchor_3.weight)

    def _resolve_anchors(self, x, ae_anchors=None):
        if ae_anchors is not None:
            if len(ae_anchors) != 3:
                raise ValueError("ae_anchors must contain three tensors: z_ae1, z_ae2, z_ae3.")
            return ae_anchors
        # Compatibility fallback for direct GCNIIStyleIGAE.forward calls.
        return (
            self.fallback_anchor_1(x),
            self.fallback_anchor_2(x),
            self.fallback_anchor_3(x),
        )

    def forward(self, x, adj, ae_anchors=None):
        anchor_1, anchor_2, anchor_3 = self._resolve_anchors(x, ae_anchors=ae_anchors)
        z = self.gnn_1(x, adj, h0=anchor_1, layer_idx=1)
        z = self.gnn_2(z, adj, h0=anchor_2, layer_idx=2)
        z_igae = self.gnn_3(z, adj, h0=anchor_3, layer_idx=3, active=False)
        z_igae_adj = self.s(torch.mm(z_igae, z_igae.t()))
        return z_igae, z_igae_adj


class FixedH0GCNIIEncoder(nn.Module):
    """GCNII encoder with a fixed hidden anchor H0 across all layers."""

    def __init__(self, hidden_dim, num_layers=4, alpha=0.2, lamda=0.5):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = max(1, int(num_layers))
        self.layers = nn.ModuleList(
            [
                GCNIIStyleLayer(
                    self.hidden_dim,
                    self.hidden_dim,
                    self.hidden_dim,
                    alpha=alpha,
                    lamda=lamda,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.last_boundary_gate_stats = {"enabled": 0}
        self.last_hidden_shape = None

    def forward(self, h0, adj):
        h = h0
        for layer_idx, layer in enumerate(self.layers, start=1):
            h = layer(
                h,
                adj,
                h0=h0,
                layer_idx=layer_idx,
                active=layer_idx < self.num_layers,
            )
        self.last_hidden_shape = tuple(h.shape)
        return h


class FixedH0GCNIIIGAE(nn.Module):
    def __init__(
        self,
        hidden_dim,
        n_z,
        gae_n_dec_1,
        gae_n_dec_2,
        gae_n_dec_3,
        n_input,
        num_layers=4,
        alpha=0.2,
        lamda=0.5,
    ):
        super().__init__()
        self.encoder = FixedH0GCNIIEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            alpha=alpha,
            lamda=lamda,
        )
        self.graph_projector = nn.Linear(int(hidden_dim), int(n_z), bias=False)
        self.decoder = SimpleIGAEDecoder(
            gae_n_dec_1=gae_n_dec_1,
            gae_n_dec_2=gae_n_dec_2,
            gae_n_dec_3=gae_n_dec_3,
            n_input=n_input,
        )
        nn.init.xavier_uniform_(self.graph_projector.weight)

    def project_graph(self, h_graph):
        return self.graph_projector(h_graph)

    def decode_hidden(self, h_graph, adj):
        return self.decoder(self.project_graph(h_graph), adj)


class GCNIIStyleIGAE(nn.Module):
    def __init__(
        self,
        gae_n_enc_1,
        gae_n_enc_2,
        gae_n_enc_3,
        gae_n_dec_1,
        gae_n_dec_2,
        gae_n_dec_3,
        n_input,
        ae_n_enc_1=None,
        ae_n_enc_2=None,
        ae_n_enc_3=None,
        alpha=0.1,
        lamda=0.5,
    ):
        super().__init__()
        self.encoder = GCNIIStyleEncoder(
            gae_n_enc_1=gae_n_enc_1,
            gae_n_enc_2=gae_n_enc_2,
            gae_n_enc_3=gae_n_enc_3,
            n_input=n_input,
            ae_n_enc_1=ae_n_enc_1,
            ae_n_enc_2=ae_n_enc_2,
            ae_n_enc_3=ae_n_enc_3,
            alpha=alpha,
            lamda=lamda,
        )
        self.decoder = SimpleIGAEDecoder(
            gae_n_dec_1=gae_n_dec_1,
            gae_n_dec_2=gae_n_dec_2,
            gae_n_dec_3=gae_n_dec_3,
            n_input=n_input,
        )

    def forward(self, x, adj, ae_anchors=None):
        z_igae, z_igae_adj = self.encoder(x, adj, ae_anchors=ae_anchors)
        z_hat, z_hat_adj = self.decoder(z_igae, adj)
        adj_hat = z_igae_adj + z_hat_adj
        return z_igae, z_hat, adj_hat
