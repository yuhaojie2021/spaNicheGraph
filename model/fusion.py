import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


def resolve_fusion_output_dim(latent_dim, fusion_mode):
    fusion_mode = str(fusion_mode).lower()
    if fusion_mode == "attention":
        return int(latent_dim)
    return int(latent_dim) * 2


class ModalityGate(nn.Module):
    """Dimension-wise gate for two modality embeddings."""

    def __init__(self, latent_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = int(latent_dim) if hidden_dim is None else int(hidden_dim)
        latent_dim = int(latent_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, emb1, emb2):
        gate_logit = self.net(torch.cat([emb1, emb2], dim=1))
        alpha = torch.sigmoid(gate_logit)
        weighted_emb1 = alpha * emb1
        weighted_emb2 = (1.0 - alpha) * emb2
        fused = torch.cat([weighted_emb1, weighted_emb2], dim=1)
        spot_weights = torch.cat(
            [alpha.mean(dim=1, keepdim=True), (1.0 - alpha).mean(dim=1, keepdim=True)],
            dim=1,
        )
        return fused, spot_weights, alpha


class AttentionLayer(nn.Module):
    """Attention fusion for two modality embeddings."""

    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.w_omega = Parameter(torch.FloatTensor(int(in_feat), int(out_feat)))
        self.u_omega = Parameter(torch.FloatTensor(int(out_feat), 1))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)

    def forward(self, emb1, emb2):
        emb = torch.stack([emb1, emb2], dim=1)
        v = torch.tanh(torch.matmul(emb, self.w_omega))
        vu = torch.matmul(v, self.u_omega).squeeze(-1)
        alpha = F.softmax(vu, dim=1)
        emb_combined = torch.sum(emb * alpha.unsqueeze(-1), dim=1)
        return emb_combined, alpha
