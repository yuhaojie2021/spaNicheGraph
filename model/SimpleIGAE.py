import torch
from torch import nn


def _graph_mm(adj, features):
    if adj.is_sparse:
        return torch.spmm(adj, features)
    return torch.matmul(adj, features)


class SimpleGCNLayer(nn.Module):
    """Single first-order graph convolution layer."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(int(in_features), int(out_features)))
        self.act = nn.Tanh()
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):
        output = torch.mm(features, self.weight)
        if active:
            output = self.act(output)
        return _graph_mm(adj, output)


class SimpleIGAEEncoder(nn.Module):
    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, n_input):
        super().__init__()
        self.gnn_1 = SimpleGCNLayer(n_input, gae_n_enc_1)
        self.gnn_2 = SimpleGCNLayer(gae_n_enc_1, gae_n_enc_2)
        self.gnn_3 = SimpleGCNLayer(gae_n_enc_2, gae_n_enc_3)
        self.s = nn.Sigmoid()
        self.last_boundary_gate_stats = {"enabled": 0}

    def forward(self, x, adj):
        z = self.gnn_1(x, adj)
        z = self.gnn_2(z, adj)
        z_igae = self.gnn_3(z, adj, active=False)
        z_igae_adj = self.s(torch.mm(z_igae, z_igae.t()))
        return z_igae, z_igae_adj


class SimpleIGAEDecoder(nn.Module):
    def __init__(self, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input):
        super().__init__()
        self.gnn_4 = SimpleGCNLayer(gae_n_dec_1, gae_n_dec_2)
        self.gnn_5 = SimpleGCNLayer(gae_n_dec_2, gae_n_dec_3)
        self.gnn_6 = SimpleGCNLayer(gae_n_dec_3, n_input)
        self.s = nn.Sigmoid()

    def forward(self, z_igae, adj):
        z = self.gnn_4(z_igae, adj)
        z = self.gnn_5(z, adj)
        z_hat = self.gnn_6(z, adj, active=False)
        z_hat_adj = self.s(torch.mm(z_hat, z_hat.t()))
        return z_hat, z_hat_adj


class SimpleIGAE(nn.Module):
    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input):
        super().__init__()
        self.encoder = SimpleIGAEEncoder(gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, n_input)
        self.decoder = SimpleIGAEDecoder(gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input)

    def forward(self, x, adj):
        z_igae, z_igae_adj = self.encoder(x, adj)
        z_hat, z_hat_adj = self.decoder(z_igae, adj)
        return z_igae, z_hat, z_igae_adj + z_hat_adj
