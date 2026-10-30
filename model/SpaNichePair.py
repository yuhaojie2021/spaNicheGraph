import torch
import torch.nn.functional as F
from torch import nn

from model.AE import AE, AE_encoder, AE_decoder
from model.GCNIIIGAE import FixedH0GCNIIIGAE, GCNIIStyleIGAE
from model.SimpleIGAE import SimpleIGAE
from model.fusion import AttentionLayer, ModalityGate, resolve_fusion_output_dim


class ProjectedAttributeAE(nn.Module):
    """AE branch that keeps a wider attribute anchor and projects it for decoding."""

    def __init__(
        self,
        ae_n_enc_1,
        ae_n_enc_2,
        ae_n_enc_3,
        ae_n_dec_1,
        ae_n_dec_2,
        ae_n_dec_3,
        n_input,
        n_z,
        graph_hidden_dim,
    ):
        super().__init__()
        self.encoder = AE_encoder(
            ae_n_enc_1=ae_n_enc_1,
            ae_n_enc_2=ae_n_enc_2,
            ae_n_enc_3=ae_n_enc_3,
            n_input=n_input,
            n_z=graph_hidden_dim,
        )
        self.attr_projector = nn.Linear(int(graph_hidden_dim), int(n_z), bias=False)
        self.decoder = AE_decoder(
            ae_n_dec_1=ae_n_dec_1,
            ae_n_dec_2=ae_n_dec_2,
            ae_n_dec_3=ae_n_dec_3,
            n_input=n_input,
            n_z=n_z,
        )
        nn.init.xavier_uniform_(self.attr_projector.weight)

    def project_attribute(self, h0):
        return self.attr_projector(h0)

    def decode_hidden(self, h0):
        return self.decoder(self.project_attribute(h0))

    def forward(self, x):
        z1, z2, h0 = self.encoder(x)
        z_attr = self.project_attribute(h0)
        x_hat = self.decoder(z_attr)
        return z1, z2, z_attr, x_hat


class SpaNichePair(nn.Module):
    """Domain-only spaNichePair backbone.

    The model contains two modality-specific AE branches and two graph branches.
    Graph propagation is anchored by AE attributes, then both modality latents are
    fused for spatial domain identification.
    """

    def __init__(
        self,
        ae_n_enc_1,
        ae_n_enc_2,
        ae_n_enc_3,
        ae_n_dec_1,
        ae_n_dec_2,
        ae_n_dec_3,
        gae_n_enc_1,
        gae_n_enc_2,
        gae_n_enc_3,
        gae_n_dec_1,
        gae_n_dec_2,
        gae_n_dec_3,
        n_input,
        n_input1,
        n_z,
        n_clusters,
        sigma,
        v=1.0,
        n_node=None,
        device=None,
        fusion_mode="concat",
        fusion_hidden_dim=None,
        graph_encoder_variant="gcnii",
        gcnii_alpha=0.1,
        gcnii_lambda=0.5,
        graph_hidden_dim=64,
        gcnii_num_layers=4,
        use_boundary_aware_gcn=0,
        boundary_gate_temperature=0.5,
        boundary_gate_min=0.2,
    ):
        super().__init__()
        self.s = nn.Sigmoid()
        self.v = v
        self.sigma = float(sigma)
        self.n_z = int(n_z)
        self.n_clusters = int(n_clusters)
        self.n_input = int(n_input)
        self.n_input1 = int(n_input1)
        self.graph_encoder_variant = str(graph_encoder_variant).lower()
        self.gcnii_alpha = float(gcnii_alpha)
        self.gcnii_lambda = float(gcnii_lambda)
        self.graph_hidden_dim = max(1, int(graph_hidden_dim))
        self.gcnii_num_layers = max(1, int(gcnii_num_layers))
        self.use_boundary_aware_gcn = bool(int(use_boundary_aware_gcn))
        self.boundary_gate_temperature = max(float(boundary_gate_temperature), 1e-6)
        self.boundary_gate_min = min(max(float(boundary_gate_min), 0.0), 1.0)
        self.fusion_mode = str(fusion_mode).lower()
        self.fusion_hidden_dim = fusion_hidden_dim

        if self.graph_encoder_variant == "gcnii_fixed_h0":
            fusion_latent_dim = self.graph_hidden_dim
            self.ae = self._build_projected_ae(
                ae_n_enc_1, ae_n_enc_2, ae_n_enc_3, ae_n_dec_1, ae_n_dec_2, ae_n_dec_3, n_input, n_z
            )
            self.ae1 = self._build_projected_ae(
                ae_n_enc_1, ae_n_enc_2, ae_n_enc_3, ae_n_dec_1, ae_n_dec_2, ae_n_dec_3, n_input1, n_z
            )
        else:
            fusion_latent_dim = self.n_z
            self.ae = AE(
                ae_n_enc_1=ae_n_enc_1,
                ae_n_enc_2=ae_n_enc_2,
                ae_n_enc_3=ae_n_enc_3,
                ae_n_dec_1=ae_n_dec_1,
                ae_n_dec_2=ae_n_dec_2,
                ae_n_dec_3=ae_n_dec_3,
                n_input=n_input,
                n_z=n_z,
            )
            self.ae1 = AE(
                ae_n_enc_1=ae_n_enc_1,
                ae_n_enc_2=ae_n_enc_2,
                ae_n_enc_3=ae_n_enc_3,
                ae_n_dec_1=ae_n_dec_1,
                ae_n_dec_2=ae_n_dec_2,
                ae_n_dec_3=ae_n_dec_3,
                n_input=n_input1,
                n_z=n_z,
            )

        self.fusion_output_dim = resolve_fusion_output_dim(fusion_latent_dim, self.fusion_mode)
        self.atten_cross = AttentionLayer(fusion_latent_dim, fusion_latent_dim)
        self.gate_fusion = ModalityGate(fusion_latent_dim, fusion_hidden_dim)

        self.gae = self._build_graph_branch(
            n_input=n_input,
            ae_n_enc_1=ae_n_enc_1,
            ae_n_enc_2=ae_n_enc_2,
            ae_n_enc_3=ae_n_enc_3,
            gae_n_enc_1=gae_n_enc_1,
            gae_n_enc_2=gae_n_enc_2,
            gae_n_enc_3=gae_n_enc_3,
            gae_n_dec_1=gae_n_dec_1,
            gae_n_dec_2=gae_n_dec_2,
            gae_n_dec_3=gae_n_dec_3,
        )
        self.gae1 = self._build_graph_branch(
            n_input=n_input1,
            ae_n_enc_1=ae_n_enc_1,
            ae_n_enc_2=ae_n_enc_2,
            ae_n_enc_3=ae_n_enc_3,
            gae_n_enc_1=gae_n_enc_1,
            gae_n_enc_2=gae_n_enc_2,
            gae_n_enc_3=gae_n_enc_3,
            gae_n_dec_1=gae_n_dec_1,
            gae_n_dec_2=gae_n_dec_2,
            gae_n_dec_3=gae_n_dec_3,
        )

        self.last_fusion = None
        self.last_branch_state = None
        self.last_dynamic_graph = None

    def _build_projected_ae(
        self,
        ae_n_enc_1,
        ae_n_enc_2,
        ae_n_enc_3,
        ae_n_dec_1,
        ae_n_dec_2,
        ae_n_dec_3,
        n_input,
        n_z,
    ):
        return ProjectedAttributeAE(
            ae_n_enc_1=ae_n_enc_1,
            ae_n_enc_2=ae_n_enc_2,
            ae_n_enc_3=ae_n_enc_3,
            ae_n_dec_1=ae_n_dec_1,
            ae_n_dec_2=ae_n_dec_2,
            ae_n_dec_3=ae_n_dec_3,
            n_input=n_input,
            n_z=n_z,
            graph_hidden_dim=self.graph_hidden_dim,
        )

    def _build_graph_branch(
        self,
        n_input,
        ae_n_enc_1,
        ae_n_enc_2,
        ae_n_enc_3,
        gae_n_enc_1,
        gae_n_enc_2,
        gae_n_enc_3,
        gae_n_dec_1,
        gae_n_dec_2,
        gae_n_dec_3,
    ):
        if self.graph_encoder_variant == "simplegcn":
            return SimpleIGAE(gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input)
        if self.graph_encoder_variant == "gcnii":
            return GCNIIStyleIGAE(
                gae_n_enc_1=gae_n_enc_1,
                gae_n_enc_2=gae_n_enc_2,
                gae_n_enc_3=gae_n_enc_3,
                gae_n_dec_1=gae_n_dec_1,
                gae_n_dec_2=gae_n_dec_2,
                gae_n_dec_3=gae_n_dec_3,
                n_input=n_input,
                ae_n_enc_1=ae_n_enc_1,
                ae_n_enc_2=ae_n_enc_2,
                ae_n_enc_3=ae_n_enc_3,
                alpha=self.gcnii_alpha,
                lamda=self.gcnii_lambda,
            )
        if self.graph_encoder_variant == "gcnii_fixed_h0":
            return FixedH0GCNIIIGAE(
                hidden_dim=self.graph_hidden_dim,
                n_z=self.n_z,
                gae_n_dec_1=gae_n_dec_1,
                gae_n_dec_2=gae_n_dec_2,
                gae_n_dec_3=gae_n_dec_3,
                n_input=n_input,
                num_layers=self.gcnii_num_layers,
                alpha=self.gcnii_alpha,
                lamda=self.gcnii_lambda,
            )
        raise ValueError(f"Unsupported graph_encoder_variant: {self.graph_encoder_variant}")

    def _decode_branch(self, ae, gae, z_ae, z_tilde, z_igae_adj, adj):
        if self.graph_encoder_variant == "gcnii_fixed_h0":
            x_hat = ae.decode_hidden(z_ae)
            z_hat, z_hat_adj = gae.decode_hidden(z_tilde, adj)
        else:
            x_hat = ae.decoder(z_ae)
            z_hat, z_hat_adj = gae.decoder(z_tilde, adj)
        adj_hat = z_igae_adj + z_hat_adj
        return x_hat, z_hat, adj_hat

    def _boundary_gate_adjacency(self, adj, attribute_anchor, gae):
        if not self.use_boundary_aware_gcn:
            if hasattr(gae.encoder, "last_boundary_gate_stats"):
                gae.encoder.last_boundary_gate_stats = {"enabled": 0}
            return adj

        anchor = F.normalize(attribute_anchor.detach(), p=2, dim=1)
        min_gate = self.boundary_gate_min
        temperature = self.boundary_gate_temperature

        if adj.is_sparse:
            adj = adj.coalesce()
            indices = adj.indices()
            values = adj.values()
            row, col = indices[0], indices[1]
            non_self = row != col
            gate = torch.ones_like(values)
            if bool(non_self.any().item()):
                cosine = (anchor[row[non_self]] * anchor[col[non_self]]).sum(dim=1)
                edge_gate = min_gate + (1.0 - min_gate) * torch.sigmoid(cosine / temperature)
                gate[non_self] = edge_gate.to(dtype=values.dtype)
            gated_adj = torch.sparse_coo_tensor(indices, values * gate, adj.shape, device=adj.device, dtype=adj.dtype).coalesce()
            stat_values = gate[non_self].detach()
        else:
            similarity = torch.matmul(anchor, anchor.t())
            gate = min_gate + (1.0 - min_gate) * torch.sigmoid(similarity / temperature)
            diag_idx = torch.arange(gate.shape[0], device=gate.device)
            gate[diag_idx, diag_idx] = 1.0
            support = adj != 0
            gated_adj = adj * gate.to(dtype=adj.dtype)
            eye = torch.eye(gate.shape[0], dtype=torch.bool, device=gate.device)
            stat_values = gate[(support & ~eye)].detach()

        if stat_values.numel() > 0:
            stats = {
                "enabled": 1,
                "edge_count": int(stat_values.numel()),
                "mean": float(stat_values.mean().item()),
                "std": float(stat_values.std(unbiased=False).item()),
                "min": float(stat_values.min().item()),
                "max": float(stat_values.max().item()),
                "temperature": float(temperature),
                "min_gate": float(min_gate),
            }
        else:
            stats = {
                "enabled": 1,
                "edge_count": 0,
                "mean": 1.0,
                "std": 0.0,
                "min": 1.0,
                "max": 1.0,
                "temperature": float(temperature),
                "min_gate": float(min_gate),
            }
        gae.encoder.last_boundary_gate_stats = stats
        return gated_adj

    def _encode_branch(self, ae, gae, inputs, adj):
        sigma = self.sigma
        if self.graph_encoder_variant == "simplegcn":
            z_ae1, z_ae2, z_ae3 = ae.encoder(inputs)
            propagation_adj = self._boundary_gate_adjacency(adj, z_ae3, gae)
            z_igae1 = gae.encoder.gnn_1(inputs, propagation_adj)
            z_igae2 = gae.encoder.gnn_2((1.0 - sigma) * z_ae1 + sigma * z_igae1, propagation_adj)
            z_igae3 = gae.encoder.gnn_3((1.0 - sigma) * z_ae2 + sigma * z_igae2, propagation_adj, active=False)
            z_tilde = (1.0 - sigma) * z_ae3 + sigma * z_igae3
            return z_ae3, z_igae3, z_tilde, self.s(torch.mm(z_tilde, z_tilde.t()))

        if self.graph_encoder_variant == "gcnii":
            z_ae1, z_ae2, z_ae3 = ae.encoder(inputs)
            propagation_adj = self._boundary_gate_adjacency(adj, z_ae3, gae)
            z_igae1 = gae.encoder.gnn_1(inputs, propagation_adj, h0=z_ae1, layer_idx=1)
            z_igae2 = gae.encoder.gnn_2(z_igae1, propagation_adj, h0=z_ae2, layer_idx=2)
            z_igae3 = gae.encoder.gnn_3(z_igae2, propagation_adj, h0=z_ae3, layer_idx=3, active=False)
            z_tilde = (1.0 - sigma) * z_ae3 + sigma * z_igae3
            return z_ae3, z_igae3, z_tilde, self.s(torch.mm(z_tilde, z_tilde.t()))

        if self.graph_encoder_variant == "gcnii_fixed_h0":
            _, _, h0 = ae.encoder(inputs)
            propagation_adj = self._boundary_gate_adjacency(adj, h0, gae)
            h_graph = gae.encoder(h0, propagation_adj)
            z_tilde = (1.0 - sigma) * h0 + sigma * h_graph
            return h0, h_graph, z_tilde, self.s(torch.mm(z_tilde, z_tilde.t()))

        raise ValueError(f"Unsupported graph_encoder_variant: {self.graph_encoder_variant}")

    def _record_branch_state(self, z_tilde, z_tilde1):
        cosine = F.cosine_similarity(z_tilde.detach(), z_tilde1.detach(), dim=1)
        self.last_branch_state = {
            "omics1": z_tilde.detach(),
            "omics2": z_tilde1.detach(),
            "cosine_mean": float(cosine.mean().item()),
            "cosine_std": float(cosine.std(unbiased=False).item()),
        }

    def _record_fusion(self, mode, weights, gate_map=None):
        weights_cpu = weights.detach().cpu()
        weight_mean = weights_cpu.mean(dim=0)
        self.last_fusion = {
            "mode": mode,
            "weights": weights_cpu.tolist(),
            "omics1_weight_mean": float(weight_mean[0].item()),
            "omics2_weight_mean": float(weight_mean[1].item()),
        }
        if gate_map is not None:
            self.last_fusion["gate_dim_mean"] = gate_map.detach().cpu().mean(dim=0).tolist()

    def _fuse_latent(self, z_tilde, z_tilde1):
        self._record_branch_state(z_tilde, z_tilde1)
        mode = self.fusion_mode
        gate_map = None
        if mode == "concat":
            fea = torch.cat([z_tilde, z_tilde1], dim=1)
            weights = torch.full((z_tilde.size(0), 2), 0.5, device=z_tilde.device, dtype=z_tilde.dtype)
        elif mode in {"gate", "adaptive_gate", "gated"}:
            fea, weights, gate_map = self.gate_fusion(z_tilde, z_tilde1)
            mode = "adaptive_gate"
        elif mode == "attention":
            fea, weights = self.atten_cross(z_tilde, z_tilde1)
        else:
            raise ValueError(f"Unsupported fusion_mode: {self.fusion_mode}")
        self._record_fusion(mode, weights, gate_map=gate_map)
        return fea

    def init(self, X_tilde1, X_tilde2, adj1):
        _, _, z_tilde, _ = self._encode_branch(self.ae, self.gae, X_tilde1, adj1)
        _, _, z_tilde1, _ = self._encode_branch(self.ae1, self.gae1, X_tilde2, adj1)
        return self._fuse_latent(z_tilde, z_tilde1)

    def forward(self, X_tilde1, X_tilde2, adj1):
        if int(X_tilde1.shape[1]) != self.n_input:
            raise ValueError(f"omics1 input dim mismatch: got {int(X_tilde1.shape[1])}, expected {self.n_input}.")
        if int(X_tilde2.shape[1]) != self.n_input1:
            raise ValueError(f"omics2 input dim mismatch: got {int(X_tilde2.shape[1])}, expected {self.n_input1}.")
        z_ae3, z_igae3, z_tilde, z_igae_adj = self._encode_branch(self.ae, self.gae, X_tilde1, adj1)
        z_ae31, z_igae31, z_tilde1, z_igae_adj1 = self._encode_branch(self.ae1, self.gae1, X_tilde2, adj1)

        x_hat, z_hat, adj_hat = self._decode_branch(self.ae, self.gae, z_ae3, z_tilde, z_igae_adj, adj1)
        x_hat1, z_hat1, adj_hat1 = self._decode_branch(self.ae1, self.gae1, z_ae31, z_tilde1, z_igae_adj1, adj1)
        fea = self._fuse_latent(z_tilde, z_tilde1)

        return x_hat, z_hat, adj_hat, z_ae3, z_igae3, x_hat1, z_hat1, adj_hat1, z_ae31, z_igae31, fea
