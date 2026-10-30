import contextlib
import io
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from torch.optim import Adam

from train.utils import clustering
from utils.misc import eva, print_results
from utils.preprocess import blend_adjacency_matrices, preprocess_graph, update_dynamic_niche_graph


BCE_loss = torch.nn.BCEWithLogitsLoss()


def cosine_sim_tensor(emb):
    norm = torch.norm(emb, p=2, dim=1, keepdim=True).clamp_min(1e-12)
    emb_norm = emb / norm
    return torch.matmul(emb_norm, emb_norm.T)


def noise_cross_entropy(emb, adj):
    sim_exp = torch.exp(cosine_sim_tensor(emb))
    positive = torch.mul(sim_exp, adj).sum(dim=1).clamp_min(1e-12)
    negative = torch.mul(sim_exp, 1.0 - adj).sum(dim=1).clamp_min(1e-12)
    return -torch.log(positive / negative).mean()


def adjacency_to_dense_tensor(adj, ref_tensor):
    if adj is None:
        return None
    if isinstance(adj, torch.Tensor):
        dense_adj = adj.to_dense() if adj.is_sparse else adj
    elif hasattr(adj, "toarray"):
        dense_adj = torch.as_tensor(adj.toarray())
    else:
        dense_adj = torch.as_tensor(np.asarray(adj))
    return dense_adj.to(device=ref_tensor.device, dtype=ref_tensor.dtype)


def dense_adj_to_sparse_graph_tensor(dense_adj, device):
    return preprocess_graph(np.asarray(dense_adj, dtype=np.float32)).to(device)


def compute_contrastive_regularization(emb, adj_spatial=None, temperature=0.2):
    spatial_adj = adjacency_to_dense_tensor(adj_spatial, emb)
    if spatial_adj is None:
        zero_loss = torch.zeros((), device=emb.device, dtype=emb.dtype)
        return zero_loss, {"loss_nce_mode": "disabled"}
    spatial_binary = torch.where(spatial_adj > 0, 1.0, 0.0)
    loss = noise_cross_entropy(emb, spatial_binary)
    return loss, {"loss_nce_mode": "single_view_spatial"}


def resolve_contrastive_weight(model, epoch, epochs, args):
    base_weight = float(getattr(args, "loss_n", 0.0) or 0.0)
    if base_weight <= 0.0:
        return 0.0, {
            "loss_nce_phase_scale": 0.0,
            "loss_nce_alignment_scale": 0.0,
            "loss_nce_balance_scale": 0.0,
            "branch_cosine_mean": None,
            "branch_cosine_std": None,
        }

    epoch_ratio = float(epoch + 1) / float(max(epochs, 1))
    start_ratio = float(getattr(args, "contrastive_start_ratio", 0.1) or 0.0)
    ramp_ratio = float(getattr(args, "contrastive_ramp_ratio", 0.2) or 0.0)
    if epoch_ratio <= start_ratio:
        phase_scale = 0.0
    elif ramp_ratio <= 0.0:
        phase_scale = 1.0
    else:
        phase_scale = min(1.0, max(0.0, (epoch_ratio - start_ratio) / ramp_ratio))

    branch_state = getattr(model, "last_branch_state", None) or {}
    cosine_mean = branch_state.get("cosine_mean")
    cosine_std = branch_state.get("cosine_std")
    alignment_scale = 1.0
    if cosine_mean is not None:
        power = max(float(getattr(args, "contrastive_alignment_power", 1.0) or 1.0), 1e-6)
        min_scale = max(0.0, float(getattr(args, "contrastive_min_scale", 0.0) or 0.0))
        max_scale = max(min_scale, float(getattr(args, "contrastive_max_scale", 1.0) or 1.0))
        alignment_signal = min(max(1.0 - float(cosine_mean), 0.0), 1.0)
        alignment_scale = min_scale + (max_scale - min_scale) * (alignment_signal**power)

    fusion_info = getattr(model, "last_fusion", None) or {}
    mean_w1 = float(fusion_info.get("omics1_weight_mean", 0.5))
    mean_w2 = float(fusion_info.get("omics2_weight_mean", 0.5))
    balance_signal = max(0.0, 1.0 - abs(mean_w1 - mean_w2))
    balance_floor = min(max(float(getattr(args, "contrastive_balance_floor", 0.25) or 0.0), 0.0), 1.0)
    balance_scale = balance_floor + (1.0 - balance_floor) * balance_signal

    return base_weight * phase_scale * alignment_scale * balance_scale, {
        "loss_nce_phase_scale": float(phase_scale),
        "loss_nce_alignment_scale": float(alignment_scale),
        "loss_nce_balance_scale": float(balance_scale),
        "branch_cosine_mean": None if cosine_mean is None else float(cosine_mean),
        "branch_cosine_std": None if cosine_std is None else float(cosine_std),
    }


def refresh_dynamic_niche_graph(
    embedding,
    static_spatial_adj_dense,
    static_niche_adj_dense,
    niche_weight,
    n_neighbors=12,
    metric="cosine",
    smooth_beta=0.2,
    beta_min=0.02,
    adaptive_beta=True,
    alpha_min=0.05,
    adaptive_alpha=False,
):
    dynamic_niche_adj, dynamic_stats = update_dynamic_niche_graph(
        embedding,
        static_niche_adj=static_niche_adj_dense,
        n_neighbors=n_neighbors,
        metric=metric,
        smooth_beta=smooth_beta,
        beta_min=beta_min,
        adaptive_beta=adaptive_beta,
        return_stats=True,
    )
    alpha_max = min(max(float(niche_weight or 0.0), 0.0), 1.0)
    alpha_min = min(max(float(alpha_min or 0.0), 0.0), alpha_max)
    alpha_confidence = float(dynamic_stats.get("edge_overlap", 0.0))
    alpha = alpha_min + (alpha_max - alpha_min) * alpha_confidence if adaptive_alpha else alpha_max
    dynamic_stats.update(
        {
            "version": "v3_adaptive_alpha_beta" if adaptive_alpha else "v2_fixed_alpha_adaptive_beta",
            "adaptive_alpha": int(bool(adaptive_alpha)),
            "alpha": float(alpha),
            "alpha_min": float(alpha_min),
            "alpha_max": float(alpha_max),
            "alpha_confidence": float(alpha_confidence),
        }
    )
    dynamic_mixed_adj = blend_adjacency_matrices(static_spatial_adj_dense, dynamic_niche_adj, niche_weight=alpha)
    return dynamic_niche_adj, dynamic_mixed_adj, dynamic_stats


def run_model_forward(model, data, data1, adj):
    return model(data, data1, adj)


def validate_modal_input_dims(data, data1, args):
    expected_dim = int(getattr(args, "n_input"))
    expected_dim1 = int(getattr(args, "n_input1"))
    actual_dim = int(data.shape[1])
    actual_dim1 = int(data1.shape[1])
    if actual_dim != expected_dim:
        raise ValueError(f"omics1 input dim mismatch: data.shape[1]={actual_dim}, args.n_input={expected_dim}.")
    if actual_dim1 != expected_dim1:
        raise ValueError(f"omics2 input dim mismatch: data1.shape[1]={actual_dim1}, args.n_input1={expected_dim1}.")


def compute_unsupervised_cluster_score(embedding, pred_labels, metric="silhouette", sample_size=2000, random_seed=0):
    if embedding is None or pred_labels is None:
        return None
    embedding_np = np.asarray(embedding, dtype=np.float32)
    pred_labels = np.asarray(pred_labels)
    if embedding_np.ndim != 2 or pred_labels.ndim != 1:
        return None
    n_samples = int(embedding_np.shape[0])
    n_labels = int(np.unique(pred_labels).size)
    if n_samples < 3 or n_labels < 2 or n_labels >= n_samples:
        return None

    metric = str(metric or "silhouette").lower()
    sample_size = int(sample_size or 0)
    try:
        if metric in {"silhouette", "sil"}:
            if sample_size <= 0 or sample_size >= n_samples:
                return float(silhouette_score(embedding_np, pred_labels, metric="euclidean"))
            return float(
                silhouette_score(
                    embedding_np,
                    pred_labels,
                    metric="euclidean",
                    sample_size=sample_size,
                    random_state=int(random_seed),
                )
            )
        if metric in {"calinski", "calinski_harabasz", "ch"}:
            return float(calinski_harabasz_score(embedding_np, pred_labels))
        if metric in {"davies", "davies_bouldin", "db"}:
            return -float(davies_bouldin_score(embedding_np, pred_labels))
    except ValueError:
        return None
    raise ValueError(f"Unsupported unsupervised early-stop metric: {metric}")


def cluster_embedding(embedding_np, args, tool="kmeans", adata=None, add_key="spaNicheGraph"):
    tool_name = str(tool or "kmeans").lower()
    embedding_np = np.asarray(embedding_np, dtype=np.float32)
    if tool_name == "kmeans":
        return KMeans(n_clusters=args.n_clusters, n_init=20, random_state=int(args.random_seed)).fit(embedding_np).labels_
    if tool_name == "gmm":
        return GaussianMixture(
            n_components=args.n_clusters,
            covariance_type="full",
            random_state=int(args.random_seed),
            n_init=10,
        ).fit_predict(embedding_np)
    if tool_name == "spectral":
        return SpectralClustering(
            n_clusters=args.n_clusters,
            random_state=int(args.random_seed),
            assign_labels="kmeans",
        ).fit_predict(embedding_np)
    if tool_name in {"leiden", "louvain", "mclust"}:
        if adata is None:
            import scanpy as sc

            adata = sc.AnnData(np.zeros((embedding_np.shape[0], 1), dtype=np.float32))
        adata.obsm["X_spaNicheGraph"] = embedding_np
        n_comps = max(2, min(20, embedding_np.shape[0] - 1, embedding_np.shape[1]))
        with contextlib.redirect_stdout(io.StringIO()):
            clustering(
                adata,
                key="X_spaNicheGraph",
                add_key=add_key,
                n_clusters=args.n_clusters,
                method=tool_name,
                use_pca=False,
                n_comps=n_comps,
            )
        return np.asarray(adata.obs[tool_name])
    raise ValueError(f"Unsupported clustering tool: {tool}")


def resolve_dynamic_graph_source(model, fea, source_label):
    source_label = str(source_label or "joint").lower()
    branch_state = getattr(model, "last_branch_state", None) or {}
    if source_label in {"omics1", "modality1"} and branch_state.get("omics1") is not None:
        return branch_state["omics1"].detach().cpu().numpy(), "omics1"
    if source_label in {"omics2", "modality2"} and branch_state.get("omics2") is not None:
        return branch_state["omics2"].detach().cpu().numpy(), "omics2"
    return fea.detach().cpu().numpy(), "joint"


def Train(
    epochs,
    model,
    data,
    data1,
    adj,
    label,
    device,
    args,
    data_target=None,
    data_target1=None,
    adj_spatial=None,
    adj_niche=None,
    dynamic_spatial_adj_dense=None,
    dynamic_niche_static_adj_dense=None,
    dynamic_mixed_static_adj_dense=None,
):
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    validate_modal_input_dims(data, data1, args)

    data_target = data if data_target is None else data_target
    data_target1 = data1 if data_target1 is None else data_target1
    optimizer = Adam(model.parameters(), lr=args.lr)
    train_history = []

    use_dynamic_niche_graph = bool(int(getattr(args, "use_dynamic_niche_graph", 0) or 0))
    dynamic_graph_warmup_epoch = int(max(0, min(epochs - 1, round(epochs * float(getattr(args, "dynamic_graph_warmup", 0.2))))))
    dynamic_graph_update_interval = max(1, int(getattr(args, "dynamic_graph_update_interval", 50) or 50))
    dynamic_graph_smooth_beta = float(getattr(args, "dynamic_graph_smooth_beta", 0.2) or 0.0)
    dynamic_graph_beta_min = float(getattr(args, "dynamic_graph_beta_min", 0.02) or 0.0)
    dynamic_graph_adaptive_beta = bool(int(getattr(args, "dynamic_graph_adaptive_beta", 1) or 0))
    dynamic_graph_alpha_min = float(getattr(args, "dynamic_graph_alpha_min", 0.05) or 0.0)
    dynamic_graph_adaptive_alpha = bool(int(getattr(args, "dynamic_graph_adaptive_alpha", 0) or 0))
    dynamic_graph_metric = str(getattr(args, "dynamic_graph_metric", "cosine") or "cosine")
    dynamic_graph_source_label = str(getattr(args, "dynamic_graph_source", "joint") or "joint")
    dynamic_graph_neighbors = max(1, int(getattr(args, "dynamic_graph_neighbors", 12) or 12))
    dynamic_graph_niche_weight = float(getattr(args, "dynamic_graph_niche_weight", 0.25) or 0.0)
    dynamic_graph_version_name = "v3_adaptive_alpha_beta" if dynamic_graph_adaptive_alpha else "v2_fixed_alpha_adaptive_beta"

    current_dynamic_niche_dense = None
    current_dynamic_mixed_dense = None
    if use_dynamic_niche_graph and dynamic_spatial_adj_dense is not None and dynamic_niche_static_adj_dense is not None:
        current_dynamic_niche_dense = np.asarray(dynamic_niche_static_adj_dense, dtype=np.float32).copy()
        current_dynamic_mixed_dense = (
            np.asarray(dynamic_mixed_static_adj_dense, dtype=np.float32).copy()
            if dynamic_mixed_static_adj_dense is not None
            else blend_adjacency_matrices(dynamic_spatial_adj_dense, current_dynamic_niche_dense, niche_weight=dynamic_graph_niche_weight)
        )
    else:
        use_dynamic_niche_graph = False

    dynamic_updates_applied = 0
    last_dynamic_update_stats = {}
    model.last_dynamic_graph = {
        "enabled": bool(use_dynamic_niche_graph),
        "warmup_epoch": int(dynamic_graph_warmup_epoch),
        "update_interval": int(dynamic_graph_update_interval),
        "smooth_beta": float(dynamic_graph_smooth_beta),
        "alpha_max": float(dynamic_graph_niche_weight),
        "alpha_min": float(dynamic_graph_alpha_min),
        "adaptive_alpha": int(dynamic_graph_adaptive_alpha),
        "beta_max": float(dynamic_graph_smooth_beta),
        "beta_min": float(dynamic_graph_beta_min),
        "adaptive_beta": int(dynamic_graph_adaptive_beta),
        "version": dynamic_graph_version_name,
        "graph_source": dynamic_graph_source_label,
        "metric": dynamic_graph_metric,
        "updates_applied": 0,
        "mixed_adj_dense": None if current_dynamic_mixed_dense is None else current_dynamic_mixed_dense.copy(),
        "niche_adj_dense": None if current_dynamic_niche_dense is None else current_dynamic_niche_dense.copy(),
    }

    early_stop_enabled = bool(int(getattr(args, "early_stop_unsupervised", 0) or 0))
    early_stop_metric = str(getattr(args, "early_stop_metric", "silhouette") or "silhouette")
    early_stop_tool = str(getattr(args, "early_stop_tool", "kmeans") or "kmeans").lower()
    train_eval_interval = max(1, int(getattr(args, "train_eval_interval", 10) or 10))
    early_stop_warmup_epoch = int(max(0, min(epochs, round(epochs * float(getattr(args, "early_stop_warmup_ratio", 0.25))))))
    early_stop_patience = max(1, int(getattr(args, "early_stop_patience", 8) or 8))
    early_stop_min_delta = max(0.0, float(getattr(args, "early_stop_min_delta", 1e-4) or 0.0))
    early_stop_sample_size = int(getattr(args, "early_stop_sample_size", 2000) or 0)
    early_stop_best_score = -np.inf
    early_stop_best_epoch = None
    early_stop_bad_count = 0
    model.early_stop_state = {
        "enabled": int(early_stop_enabled),
        "metric": early_stop_metric,
        "tool": early_stop_tool,
        "train_eval_interval": int(train_eval_interval),
        "warmup_epoch": int(early_stop_warmup_epoch),
        "patience": int(early_stop_patience),
        "min_delta": float(early_stop_min_delta),
        "sample_size": int(early_stop_sample_size),
        "best_score": None,
        "best_epoch": None,
        "stopped": 0,
        "stop_epoch": None,
        "bad_count": 0,
    }

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        x_hat, z_hat, adj_hat, z_ae, z_igae, x_hat1, z_hat1, adj_hat1, z_ae1, z_igae1, fea = run_model_forward(
            model, data, data1, adj
        )

        adj_logits = torch.mm(fea, fea.T)
        adj_reconstruct_dense = adjacency_to_dense_tensor(adj, fea)
        adj_spatial_dense = adjacency_to_dense_tensor(adj_spatial if adj_spatial is not None else adj, fea)

        loss_adj = BCE_loss(adj_logits, adj_reconstruct_dense)
        loss_nce, contrastive_stats = compute_contrastive_regularization(
            fea,
            adj_spatial=adj_spatial_dense,
            temperature=float(getattr(args, "contrastive_temperature", 0.2) or 0.2),
        )
        loss_ae = F.mse_loss(x_hat, data_target) + F.mse_loss(x_hat1, data_target1)
        loss_w = F.mse_loss(z_hat, data_target) + F.mse_loss(z_hat1, data_target1)
        loss_s = F.mse_loss(z_igae, z_ae) + F.mse_loss(z_igae1, z_ae1)
        current_nce_weight, nce_runtime_stats = resolve_contrastive_weight(model, epoch, epochs, args)
        loss_igae = args.loss_w * loss_w + args.loss_a * loss_adj
        loss = loss_ae + loss_igae + args.loss_s * loss_s
        if current_nce_weight > 0.0:
            loss = loss + current_nce_weight * loss_nce

        loss.backward()
        optimizer.step()

        history_row = {
            "epoch": epoch + 1,
            "loss_total": float(loss.detach().cpu().item()),
            "loss_ae": float(loss_ae.detach().cpu().item()),
            "loss_w": float(loss_w.detach().cpu().item()),
            "loss_igae": float(loss_igae.detach().cpu().item()),
            "loss_adj": float(loss_adj.detach().cpu().item()),
            "loss_s": float(loss_s.detach().cpu().item()),
            "loss_nce": float(loss_nce.detach().cpu().item()),
            "loss_nce_weight": float(current_nce_weight),
            "loss_nce_mode": contrastive_stats.get("loss_nce_mode", "disabled"),
            "loss_nce_phase_scale": float(nce_runtime_stats["loss_nce_phase_scale"]),
            "loss_nce_alignment_scale": float(nce_runtime_stats["loss_nce_alignment_scale"]),
            "loss_nce_balance_scale": float(nce_runtime_stats["loss_nce_balance_scale"]),
            "dynamic_graph_enabled": int(use_dynamic_niche_graph),
            "dynamic_graph_updated": 0,
            "dynamic_graph_niche_delta_fro": 0.0,
            "dynamic_graph_mixed_delta_fro": 0.0,
            "dynamic_graph_version": dynamic_graph_version_name if use_dynamic_niche_graph else "",
            "dynamic_graph_source": dynamic_graph_source_label if use_dynamic_niche_graph else "",
            "dynamic_graph_alpha": np.nan,
            "dynamic_graph_beta": np.nan,
            "dynamic_graph_edge_overlap": np.nan,
            "dynamic_graph_edge_jaccard": np.nan,
            "early_stop_enabled": int(early_stop_enabled),
            "early_stop_metric": early_stop_metric if early_stop_enabled else "",
            "early_stop_tool": early_stop_tool,
            "train_eval_interval": int(train_eval_interval),
            "early_stop_score": np.nan,
            "early_stop_best_score": np.nan,
            "early_stop_bad_count": int(early_stop_bad_count),
            "early_stop_triggered": 0,
        }
        if nce_runtime_stats["branch_cosine_mean"] is not None:
            history_row["branch_cosine_mean"] = float(nce_runtime_stats["branch_cosine_mean"])
            history_row["branch_cosine_std"] = float(nce_runtime_stats["branch_cosine_std"])

        if epoch % train_eval_interval == 0:
            print(f"{epoch:3d} loss: {float(loss.detach().cpu().item())}")
            with torch.no_grad():
                eval_embedding = fea.detach().cpu().numpy()
                eval_labels = cluster_embedding(eval_embedding, args, tool=early_stop_tool)
                if eval_labels is not None and label is not None:
                    nmi, ari, ami, homogeneity, completeness, v_measure = eva(label, eval_labels, epoch)
                    history_row.update(
                        {
                            "train_eval_tool": early_stop_tool,
                            "train_nmi": float(nmi),
                            "train_ari": float(ari),
                            "train_ami": float(ami),
                            "train_homogeneity": float(homogeneity),
                            "train_completeness": float(completeness),
                            "train_v_measure": float(v_measure),
                        }
                    )
                if early_stop_enabled and eval_labels is not None:
                    early_stop_score = compute_unsupervised_cluster_score(
                        eval_embedding,
                        eval_labels,
                        metric=early_stop_metric,
                        sample_size=early_stop_sample_size,
                        random_seed=int(args.random_seed),
                    )
                    if early_stop_score is not None:
                        improved = early_stop_score > early_stop_best_score + early_stop_min_delta
                        if improved:
                            early_stop_best_score = float(early_stop_score)
                            early_stop_best_epoch = int(epoch + 1)
                            early_stop_bad_count = 0
                        elif (epoch + 1) >= early_stop_warmup_epoch:
                            early_stop_bad_count += 1
                        history_row.update(
                            {
                                "early_stop_score": float(early_stop_score),
                                "early_stop_best_score": float(early_stop_best_score),
                                "early_stop_bad_count": int(early_stop_bad_count),
                            }
                        )
                        model.early_stop_state.update(
                            {
                                "best_score": None if not np.isfinite(early_stop_best_score) else float(early_stop_best_score),
                                "best_epoch": early_stop_best_epoch,
                                "bad_count": int(early_stop_bad_count),
                                "last_score": float(early_stop_score),
                            }
                        )
                        if (epoch + 1) >= early_stop_warmup_epoch and early_stop_bad_count >= early_stop_patience:
                            history_row["early_stop_triggered"] = 1
                            model.early_stop_state.update({"stopped": 1, "stop_epoch": int(epoch + 1)})
                            train_history.append(history_row)
                            print(
                                f"Early stop at epoch {epoch + 1}: {early_stop_metric} ({early_stop_tool}) "
                                f"did not improve for {early_stop_bad_count} evaluations after warmup epoch {early_stop_warmup_epoch}."
                            )
                            break

        if (
            use_dynamic_niche_graph
            and current_dynamic_niche_dense is not None
            and epoch >= dynamic_graph_warmup_epoch
            and ((epoch + 1 - dynamic_graph_warmup_epoch) % dynamic_graph_update_interval == 0)
        ):
            prev_niche_dense = current_dynamic_niche_dense.copy()
            prev_mixed_dense = current_dynamic_mixed_dense.copy()
            graph_update_features, graph_source_for_update = resolve_dynamic_graph_source(model, fea, dynamic_graph_source_label)
            updated_niche_dense, updated_mixed_dense, dynamic_update_stats = refresh_dynamic_niche_graph(
                graph_update_features,
                static_spatial_adj_dense=dynamic_spatial_adj_dense,
                static_niche_adj_dense=dynamic_niche_static_adj_dense,
                niche_weight=dynamic_graph_niche_weight,
                n_neighbors=dynamic_graph_neighbors,
                metric=dynamic_graph_metric,
                smooth_beta=dynamic_graph_smooth_beta,
                beta_min=dynamic_graph_beta_min,
                adaptive_beta=dynamic_graph_adaptive_beta,
                alpha_min=dynamic_graph_alpha_min,
                adaptive_alpha=dynamic_graph_adaptive_alpha,
            )
            dynamic_update_stats["graph_source"] = graph_source_for_update
            current_dynamic_niche_dense = updated_niche_dense
            current_dynamic_mixed_dense = updated_mixed_dense
            adj = dense_adj_to_sparse_graph_tensor(current_dynamic_mixed_dense, device)
            dynamic_updates_applied += 1
            last_dynamic_update_stats = dict(dynamic_update_stats)
            history_row.update(
                {
                    "dynamic_graph_updated": 1,
                    "dynamic_graph_niche_delta_fro": float(np.linalg.norm(current_dynamic_niche_dense - prev_niche_dense)),
                    "dynamic_graph_mixed_delta_fro": float(np.linalg.norm(current_dynamic_mixed_dense - prev_mixed_dense)),
                    "dynamic_graph_version": str(dynamic_update_stats.get("version", dynamic_graph_version_name)),
                    "dynamic_graph_source": str(dynamic_update_stats.get("graph_source", graph_source_for_update)),
                    "dynamic_graph_alpha": float(dynamic_update_stats.get("alpha", np.nan)),
                    "dynamic_graph_beta": float(dynamic_update_stats.get("beta", np.nan)),
                    "dynamic_graph_edge_overlap": float(dynamic_update_stats.get("edge_overlap", np.nan)),
                    "dynamic_graph_edge_jaccard": float(dynamic_update_stats.get("edge_jaccard", np.nan)),
                }
            )
            model.last_dynamic_graph = {
                "enabled": True,
                "warmup_epoch": int(dynamic_graph_warmup_epoch),
                "update_interval": int(dynamic_graph_update_interval),
                "smooth_beta": float(dynamic_graph_smooth_beta),
                "alpha_max": float(dynamic_graph_niche_weight),
                "alpha_min": float(dynamic_graph_alpha_min),
                "adaptive_alpha": int(dynamic_graph_adaptive_alpha),
                "last_alpha": float(dynamic_update_stats.get("alpha", np.nan)),
                "last_alpha_confidence": float(dynamic_update_stats.get("alpha_confidence", np.nan)),
                "beta_max": float(dynamic_graph_smooth_beta),
                "beta_min": float(dynamic_graph_beta_min),
                "adaptive_beta": int(dynamic_graph_adaptive_beta),
                "version": str(dynamic_update_stats.get("version", dynamic_graph_version_name)),
                "graph_source": str(dynamic_update_stats.get("graph_source", graph_source_for_update)),
                "last_beta": float(dynamic_update_stats.get("beta", np.nan)),
                "last_edge_overlap": float(dynamic_update_stats.get("edge_overlap", np.nan)),
                "last_edge_jaccard": float(dynamic_update_stats.get("edge_jaccard", np.nan)),
                "metric": dynamic_graph_metric,
                "updates_applied": int(dynamic_updates_applied),
                "mixed_adj_dense": current_dynamic_mixed_dense.copy(),
                "niche_adj_dense": current_dynamic_niche_dense.copy(),
                "last_niche_delta_fro": float(history_row["dynamic_graph_niche_delta_fro"]),
                "last_mixed_delta_fro": float(history_row["dynamic_graph_mixed_delta_fro"]),
            }

        train_history.append(history_row)

    if use_dynamic_niche_graph and current_dynamic_mixed_dense is not None:
        model.last_dynamic_graph.update(
            {
                "updates_applied": int(dynamic_updates_applied),
                "mixed_adj_dense": current_dynamic_mixed_dense.copy(),
                "niche_adj_dense": current_dynamic_niche_dense.copy(),
            }
        )
        if last_dynamic_update_stats:
            model.last_dynamic_graph.update(
                {
                    "last_alpha": float(last_dynamic_update_stats.get("alpha", np.nan)),
                    "last_beta": float(last_dynamic_update_stats.get("beta", np.nan)),
                    "last_edge_overlap": float(last_dynamic_update_stats.get("edge_overlap", np.nan)),
                    "last_edge_jaccard": float(last_dynamic_update_stats.get("edge_jaccard", np.nan)),
                }
            )

    model.training_history = train_history
    return model


def Test(model, adata, data, data1, adj, label, device, args, tool, data_target=None, data_target1=None, adj_spatial=None):
    nmi_result = []
    ari_result = []
    ami_result = []
    homogeneity_result = []
    completeness_result = []
    v_measure_result = []
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    validate_modal_input_dims(data, data1, args)

    model.eval()
    with torch.no_grad():
        x_hat, z_hat, adj_hat, z_ae, z_igae, x_hat1, z_hat1, adj_hat1, z_ae1, z_igae1, fea = run_model_forward(
            model, data, data1, adj
        )
        pred = cluster_embedding(fea.detach().cpu().numpy(), args, tool=tool, adata=adata.copy(), add_key="spaNicheGraph")
        pred_np = np.asarray(pred)
        adata.obs["pred"] = pd.Categorical(pred_np)
        nmi, ari, ami, homogeneity, completeness, v_measure = eva(label, pred_np)

        contrastive_summary = None
        if float(getattr(args, "loss_n", 0.0) or 0.0) > 0.0:
            _, contrastive_summary = compute_contrastive_regularization(
                fea,
                adj_spatial=adj_spatial if adj_spatial is not None else adj,
                temperature=float(getattr(args, "contrastive_temperature", 0.2) or 0.2),
            )

        nmi_result.append(nmi)
        ari_result.append(ari)
        ami_result.append(ami)
        homogeneity_result.append(homogeneity)
        completeness_result.append(completeness)
        v_measure_result.append(v_measure)
        model.last_test = {
            "embedding": fea.detach().cpu().numpy(),
            "pred": pred_np,
            "tool": tool,
            "fusion": getattr(model, "last_fusion", None),
            "reconstruction": {
                "omics1": x_hat.detach().cpu().numpy(),
                "omics2": x_hat1.detach().cpu().numpy(),
            },
            "branch_state": {
                "cosine_mean": float((getattr(model, "last_branch_state", None) or {}).get("cosine_mean", 0.0)),
                "cosine_std": float((getattr(model, "last_branch_state", None) or {}).get("cosine_std", 0.0)),
            },
            "boundary_gate": {
                "omics1": getattr(getattr(getattr(model, "gae", None), "encoder", None), "last_boundary_gate_stats", None),
                "omics2": getattr(getattr(getattr(model, "gae1", None), "encoder", None), "last_boundary_gate_stats", None),
            },
            "contrastive": contrastive_summary,
            "dynamic_graph": getattr(model, "last_dynamic_graph", None),
            "metrics": {
                "nmi": float(nmi),
                "ari": float(ari),
                "ami": float(ami),
                "homogeneity": float(homogeneity),
                "completeness": float(completeness),
                "v_measure": float(v_measure),
            },
        }

    print(f"clustering tool: {tool}")
    if nmi_result:
        print_results(nmi_result, ari_result, ami_result, args, "test")
    return nmi, ari, ami, homogeneity, completeness, v_measure


def TestUnsupervised(model, adata, data, data1, adj, device, args, tool, data_target=None, data_target1=None, adj_spatial=None):
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    validate_modal_input_dims(data, data1, args)

    model.eval()
    with torch.no_grad():
        x_hat, z_hat, adj_hat, z_ae, z_igae, x_hat1, z_hat1, adj_hat1, z_ae1, z_igae1, fea = run_model_forward(
            model, data, data1, adj
        )
        pred = cluster_embedding(fea.detach().cpu().numpy(), args, tool=tool, adata=adata.copy(), add_key="spaNicheGraph")
        pred_np = np.asarray(pred)
        adata.obs["pred"] = pd.Categorical(pred_np)

        contrastive_summary = None
        if float(getattr(args, "loss_n", 0.0) or 0.0) > 0.0:
            _, contrastive_summary = compute_contrastive_regularization(
                fea,
                adj_spatial=adj_spatial if adj_spatial is not None else adj,
                temperature=float(getattr(args, "contrastive_temperature", 0.2) or 0.2),
            )

        embedding = fea.detach().cpu().numpy()
        unsupervised_scores = {}
        for metric_name in ("silhouette", "calinski_harabasz", "davies_bouldin"):
            score = compute_unsupervised_cluster_score(
                embedding,
                pred_np,
                metric=metric_name,
                sample_size=int(getattr(args, "early_stop_sample_size", 2000) or 0),
                random_seed=int(args.random_seed),
            )
            unsupervised_scores[metric_name] = None if score is None else float(score)

        model.last_test = {
            "embedding": embedding,
            "pred": pred_np,
            "tool": tool,
            "fusion": getattr(model, "last_fusion", None),
            "reconstruction": {
                "omics1": x_hat.detach().cpu().numpy(),
                "omics2": x_hat1.detach().cpu().numpy(),
            },
            "branch_state": {
                "cosine_mean": float((getattr(model, "last_branch_state", None) or {}).get("cosine_mean", 0.0)),
                "cosine_std": float((getattr(model, "last_branch_state", None) or {}).get("cosine_std", 0.0)),
            },
            "boundary_gate": {
                "omics1": getattr(getattr(getattr(model, "gae", None), "encoder", None), "last_boundary_gate_stats", None),
                "omics2": getattr(getattr(getattr(model, "gae1", None), "encoder", None), "last_boundary_gate_stats", None),
            },
            "contrastive": contrastive_summary,
            "dynamic_graph": getattr(model, "last_dynamic_graph", None),
            "unsupervised_scores": unsupervised_scores,
        }

    print(f"clustering tool: {tool}")
    print(
        "Unsupervised scores -> "
        f"silhouette={unsupervised_scores.get('silhouette')}, "
        f"calinski_harabasz={unsupervised_scores.get('calinski_harabasz')}, "
        f"davies_bouldin={unsupervised_scores.get('davies_bouldin')}"
    )
    return unsupervised_scores
