from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from scipy.optimize import linear_sum_assignment
from torch.backends import cudnn

from model.create_model import create_model
from train.train import Test, TestUnsupervised, Train
from utils.preprocess import (
    blend_adjacency_matrices,
    build_niche_features,
    clr_normalize_each_cell,
    construct_graph_by_coordinate,
    construct_niche_graph,
    lsi,
    pca,
    preprocess_graph,
    to_numpy_array,
    transform_adjacent_matrix,
)

sc.settings.set_figure_params(facecolor="white", figsize=(5, 5))


@dataclass(frozen=True)
class DatasetDefaults:
    dataset_name: str
    data_dir: Path
    rna_filename: str
    modality_filename: str
    label_filename: str
    label_column: str
    modality_kind: str
    epoch: int
    lr: float
    sigma: float
    loss_w: float
    loss_s: float
    loss_a: float
    fusion_mode: str
    fusion_hidden_dim: int
    contrastive_start_ratio: float
    contrastive_ramp_ratio: float
    contrastive_min_scale: float
    contrastive_max_scale: float
    contrastive_alignment_power: float
    contrastive_balance_floor: float
    primary_tool: str
    secondary_tool: str | None
    spatial_neighbors: int
    niche_neighbors: int
    niche_graph_neighbors: int
    niche_graph_max_cosine_distance: float | None = None
    pair_plot_size: int = 80


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"
DEFAULT_BATCH_SEEDS = [2022, 2023, 2024, 2025, 2026]


def fix_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def create_adj_dense(adata, n_neighbors: int = 3) -> np.ndarray:
    adj_df = construct_graph_by_coordinate(adata.obsm["spatial"], n_neighbors=n_neighbors)
    adj = transform_adjacent_matrix(adj_df).toarray()
    adj = adj + adj.T
    adj = np.where(adj > 1, 1, adj)
    np.fill_diagonal(adj, 0.0)
    return adj.astype(np.float32)


def adjacency_edge_stats(adjacency: np.ndarray | None) -> dict[str, float | int | None]:
    """Return undirected edge count and density for a dense adjacency matrix."""
    if adjacency is None:
        return {"n_nodes": None, "n_edges": None, "edge_density": None}
    matrix = np.asarray(adjacency)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square adjacency matrix, got shape {matrix.shape}.")
    n_nodes = int(matrix.shape[0])
    possible = n_nodes * (n_nodes - 1) // 2
    n_edges = int(np.count_nonzero(np.triu(matrix, k=1)))
    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "edge_density": None if possible == 0 else float(n_edges / possible),
    }


def safe_n_comps(*values, minimum: int = 2) -> int:
    valid = [int(v) for v in values if v is not None and int(v) >= minimum]
    if not valid:
        raise ValueError("Cannot determine a valid PCA dimension.")
    return min(valid)


def parse_bool_option(value, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean option from value: {value!r}")


def build_args(
    dataset_name: str,
    n_input: int,
    n_input1: int,
    n_clusters: int,
    random_seed: int,
    epoch: int,
    lr: float,
    sigma: float,
    loss_n: float,
    loss_w: float,
    loss_s: float,
    loss_a: float,
    early_stop_unsupervised: int = 0,
    early_stop_tool: str = "kmeans",
    early_stop_metric: str = "silhouette",
    early_stop_warmup_ratio: float = 0.25,
    early_stop_patience: int = 8,
    early_stop_min_delta: float = 1e-4,
    early_stop_sample_size: int = 2000,
    train_eval_interval: int = 10,
    contrastive_start_ratio: float = 0.1,
    contrastive_ramp_ratio: float = 0.2,
    contrastive_min_scale: float = 0.0,
    contrastive_max_scale: float = 1.0,
    contrastive_alignment_power: float = 1.0,
    contrastive_balance_floor: float = 0.25,
    n_z: int = 40,
    model_name: str = "spanichepair_gcnii",
    gcnii_alpha: float = 0.3,
    gcnii_lambda: float = 0.5,
    graph_hidden_dim: int = 64,
    gcnii_num_layers: int = 4,
    use_boundary_aware_gcn: int = 0,
    boundary_gate_temperature: float = 0.5,
    boundary_gate_min: float = 0.2,
) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=dataset_name,
        name=model_name,
        seed=random_seed,
        cuda=torch.cuda.is_available(),
        n_input=n_input,
        n_input1=n_input1,
        n_z=int(n_z),
        freedom_degree=1.0,
        epoch=int(epoch),
        shuffle=True,
        sigma=float(sigma),
        loss_n=float(loss_n),
        loss_w=float(loss_w),
        loss_s=float(loss_s),
        loss_a=float(loss_a),
        early_stop_unsupervised=int(early_stop_unsupervised),
        early_stop_tool=early_stop_tool,
        early_stop_metric=early_stop_metric,
        early_stop_warmup_ratio=float(early_stop_warmup_ratio),
        early_stop_patience=int(early_stop_patience),
        early_stop_min_delta=float(early_stop_min_delta),
        early_stop_sample_size=int(early_stop_sample_size),
        train_eval_interval=max(1, int(train_eval_interval)),
        contrastive_start_ratio=float(contrastive_start_ratio),
        contrastive_ramp_ratio=float(contrastive_ramp_ratio),
        contrastive_min_scale=float(contrastive_min_scale),
        contrastive_max_scale=float(contrastive_max_scale),
        contrastive_alignment_power=float(contrastive_alignment_power),
        contrastive_balance_floor=float(contrastive_balance_floor),
        contrastive_temperature=0.2,
        gcnii_alpha=float(gcnii_alpha),
        gcnii_lambda=float(gcnii_lambda),
        graph_hidden_dim=int(graph_hidden_dim),
        gcnii_num_layers=int(gcnii_num_layers),
        use_boundary_aware_gcn=int(use_boundary_aware_gcn),
        boundary_gate_temperature=float(boundary_gate_temperature),
        boundary_gate_min=float(boundary_gate_min),
        use_dynamic_niche_graph=0,
        dynamic_graph_warmup=0.2,
        dynamic_graph_update_interval=50,
        dynamic_graph_smooth_beta=0.2,
        dynamic_graph_alpha_min=0.05,
        dynamic_graph_adaptive_alpha=0,
        dynamic_graph_beta_min=0.02,
        dynamic_graph_adaptive_beta=1,
        dynamic_graph_metric="cosine",
        dynamic_graph_source="joint",
        dynamic_graph_neighbors=12,
        dynamic_graph_niche_weight=0.25,
        lr=float(lr),
        acc=-1,
        f1=-1,
        n_clusters=int(n_clusters),
        random_seed=int(random_seed),
    )


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_builtin(value):
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def flatten_dict(d, parent_key: str = "", sep: str = "__"):
    items = {}
    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, sep=sep))
        else:
            items[new_key] = value
    return items


def build_run_outputs(dataset_name: str, experiment_preset: str, random_seed: int, results_root: Path | str):
    run_tag = f"{experiment_preset}_{random_seed}"
    dataset_dir = ensure_dir(Path(results_root) / dataset_name)
    preset_dir = ensure_dir(dataset_dir / experiment_preset)
    figure_dir = ensure_dir(preset_dir / "figures")
    return run_tag, preset_dir, figure_dir


def save_metrics_bundle(metrics_payload, dataset_dir: Path, run_tag: str) -> None:
    metrics_json_path = dataset_dir / f"metrics_{run_tag}.json"
    metrics_csv_path = dataset_dir / f"metrics_{run_tag}.csv"
    metrics_json_path.write_text(json.dumps(to_builtin(metrics_payload), ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([flatten_dict(to_builtin(metrics_payload))]).to_csv(metrics_csv_path, index=False, encoding="utf-8-sig")


def save_adata_safe(adata, path: Path) -> None:
    try:
        adata.write(path)
    except Exception as exc:
        print(f"Skip saving AnnData to {path}: {exc}")


def save_figure(fig, path: Path, dpi: int = 300) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {path}")


def resolve_device(device_arg: str = "auto", cuda_arg=None) -> torch.device:
    if cuda_arg is not None:
        cuda_text = str(cuda_arg).strip().lower()
        if cuda_text in {"cpu", "none"}:
            return torch.device("cpu")
        if cuda_text in {"auto", ""}:
            return resolve_device(device_arg="auto", cuda_arg=None)
        if cuda_text.startswith("cuda:"):
            cuda_text = cuda_text.split(":", 1)[1]
        if cuda_text == "cuda":
            cuda_text = "0"
        if not cuda_text.isdigit():
            raise ValueError(f"--cuda must be a CUDA index such as 0 or 1, got: {cuda_arg}")
        if not torch.cuda.is_available():
            raise RuntimeError("--cuda was set, but CUDA is not available.")
        cuda_index = int(cuda_text)
        cuda_count = torch.cuda.device_count()
        if cuda_index < 0 or cuda_index >= cuda_count:
            raise ValueError(f"Requested cuda:{cuda_index}, but only {cuda_count} CUDA device(s) are available.")
        return torch.device(f"cuda:{cuda_index}")

    if device_arg and device_arg.lower() != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        index = 1 if torch.cuda.device_count() > 1 else 0
        return torch.device(f"cuda:{index}")
    return torch.device("cpu")


def parse_presets(preset_values: list[str] | None, available_presets: dict[str, dict]) -> list[str]:
    if not preset_values:
        return list(available_presets.keys())
    parsed = []
    for value in preset_values:
        parsed.extend([item.strip() for item in value.split(",") if item.strip()])
    unknown = [name for name in parsed if name not in available_presets]
    if unknown:
        raise ValueError(f"Unknown presets: {unknown}. Available: {list(available_presets.keys())}")
    return parsed


def parse_seeds(seed_values: list[int] | None) -> list[int]:
    return list(seed_values) if seed_values else list(DEFAULT_BATCH_SEEDS)


def summarize_metrics_payload(metrics_payload: dict) -> dict:
    row = {
        "dataset": metrics_payload.get("dataset"),
        "experiment_preset": metrics_payload.get("experiment_preset"),
        "random_seed": metrics_payload.get("random_seed"),
        "model_name": metrics_payload.get("model_name"),
        "early_stop_tool": metrics_payload.get("early_stop_tool"),
        "train_eval_interval": metrics_payload.get("train_eval_interval"),
        "n_z": metrics_payload.get("n_z"),
        "graph_hidden_dim": metrics_payload.get("graph_hidden_dim"),
        "gcnii_num_layers": metrics_payload.get("gcnii_num_layers"),
        "use_boundary_aware_gcn": metrics_payload.get("use_boundary_aware_gcn"),
    }
    for key, value in (metrics_payload.get("model_input") or {}).items():
        row[f"model_input_{key}"] = value
    for omics_key, stats in (metrics_payload.get("feature_representation") or {}).items():
        if isinstance(stats, dict):
            for metric_key, value in stats.items():
                row[f"feature_representation_{omics_key}_{metric_key}"] = value
    for metric_group in ("primary", "secondary"):
        group = metrics_payload.get(metric_group) or {}
        for key, value in group.items():
            row[f"{metric_group}_{key}"] = value
    for key, value in (metrics_payload.get("branch_state") or {}).items():
        row[f"branch_state_{key}"] = value
    for key, value in (metrics_payload.get("niche_feature") or {}).items():
        row[f"niche_feature_{key}"] = value
    for key, value in (metrics_payload.get("dynamic_graph") or {}).items():
        if not isinstance(value, (dict, list)):
            row[f"dynamic_graph_{key}"] = value
    for omics_key, stats in (metrics_payload.get("boundary_gate") or {}).items():
        if isinstance(stats, dict):
            for metric_key, value in stats.items():
                row[f"boundary_gate_{omics_key}_{metric_key}"] = value
    return row


def load_existing_summary_row(metrics_path: Path) -> dict | None:
    if not metrics_path.exists():
        return None
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    return summarize_metrics_payload(payload)


def load_dataset(dataset_cfg: DatasetDefaults):
    data_dir = dataset_cfg.data_dir
    adata_omics1 = sc.read_h5ad(data_dir / dataset_cfg.rna_filename)
    adata_omics2 = sc.read_h5ad(data_dir / dataset_cfg.modality_filename)
    df_label = pd.read_csv(data_dir / dataset_cfg.label_filename, index_col=0)
    df_label.index = df_label.index.astype(str)
    if dataset_cfg.label_column not in df_label.columns:
        available_columns = ", ".join(map(str, df_label.columns))
        raise KeyError(
            f"Label column {dataset_cfg.label_column!r} not found in {data_dir / dataset_cfg.label_filename}. "
            f"Available columns: {available_columns}"
        )

    adata_omics1.var_names_make_unique()
    adata_omics2.var_names_make_unique()
    common_idx = adata_omics1.obs_names.intersection(adata_omics2.obs_names).intersection(df_label.index)
    if len(common_idx) == 0:
        raise ValueError(f"No common spots found among RNA, {dataset_cfg.modality_kind}, and labels in {data_dir}.")
    omics_already_aligned = (
        len(common_idx) == adata_omics1.n_obs == adata_omics2.n_obs
        and adata_omics1.obs_names.equals(common_idx)
        and adata_omics2.obs_names.equals(common_idx)
    )
    if not omics_already_aligned:
        adata_omics1 = adata_omics1[common_idx].copy()
        adata_omics2 = adata_omics2[common_idx].copy()
    labels_true = df_label.loc[common_idx, dataset_cfg.label_column].astype(str).to_numpy()

    print(f"spots: {adata_omics1.n_obs}")
    print(f"RNA shape: {adata_omics1.shape}")
    print(f"{dataset_cfg.modality_kind} shape: {adata_omics2.shape}")
    print(f"label classes: {len(np.unique(labels_true))}")
    return adata_omics1, adata_omics2, labels_true


def load_unlabeled_dataset(dataset_cfg: DatasetDefaults):
    data_dir = dataset_cfg.data_dir
    adata_omics1 = sc.read_h5ad(data_dir / dataset_cfg.rna_filename)
    adata_omics2 = sc.read_h5ad(data_dir / dataset_cfg.modality_filename)
    adata_omics1.var_names_make_unique()
    adata_omics2.var_names_make_unique()
    common_idx = adata_omics1.obs_names.intersection(adata_omics2.obs_names)
    if len(common_idx) == 0:
        raise ValueError(f"No common spots found between RNA and {dataset_cfg.modality_kind} in {data_dir}.")
    omics_already_aligned = (
        len(common_idx) == adata_omics1.n_obs == adata_omics2.n_obs
        and adata_omics1.obs_names.equals(common_idx)
        and adata_omics2.obs_names.equals(common_idx)
    )
    if not omics_already_aligned:
        adata_omics1 = adata_omics1[common_idx].copy()
        adata_omics2 = adata_omics2[common_idx].copy()
    if "spatial" not in adata_omics1.obsm:
        raise KeyError(f"{data_dir / dataset_cfg.rna_filename} does not contain obsm['spatial'].")
    if "spatial" not in adata_omics2.obsm:
        adata_omics2.obsm["spatial"] = np.asarray(adata_omics1.obsm["spatial"]).copy()

    print(f"spots: {adata_omics1.n_obs}")
    print(f"RNA shape: {adata_omics1.shape}")
    print(f"{dataset_cfg.modality_kind} shape: {adata_omics2.shape}")
    print("label classes: unavailable")
    return adata_omics1, adata_omics2


def filter_genes_min_cells_fast(adata, min_cells: int):
    min_cells = int(min_cells)
    if min_cells <= 0:
        return adata
    x = adata.X
    if sp.issparse(x):
        x_csr = x if sp.isspmatrix_csr(x) else x.tocsr()
        positive_counts = np.bincount(x_csr.indices[x_csr.data > 0], minlength=x_csr.shape[1])
    else:
        positive_counts = np.count_nonzero(np.asarray(x) > 0, axis=0)
    positive_counts = np.asarray(positive_counts).ravel()
    keep_mask = positive_counts >= min_cells
    adata.var["n_cells"] = positive_counts
    if not np.all(keep_mask):
        adata._inplace_subset_var(keep_mask)
    return adata


def preprocess_a1(
    adata_omics1,
    adata_omics2,
    dataset_cfg: DatasetDefaults,
    random_seed: int,
    use_niche_graph: bool,
    niche_graph_weight: float,
    niche_feature_mode: str = "mean_std",
    full_pair_niche_basis: str = "joint",
    niche_graph_neighbors: int | None = None,
    niche_graph_max_cosine_distance: float | None = None,
):
    filter_genes_min_cells_fast(adata_omics1, min_cells=10)
    sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
    rna_hvg_mask = adata_omics1.var["highly_variable"].to_numpy()
    sc.pp.normalize_total(adata_omics1, target_sum=1e4)
    sc.pp.log1p(adata_omics1)
    sc.pp.scale(adata_omics1)
    adata_omics1 = adata_omics1[:, rna_hvg_mask].copy()

    adata_omics2 = clr_normalize_each_cell(adata_omics2)
    sc.pp.scale(adata_omics2)

    rna_model_dim = safe_n_comps(20, adata_omics1.n_obs - 1, adata_omics1.n_vars - 1)
    adata_omics1.obsm["feat"] = pca(adata_omics1, n_comps=rna_model_dim)

    omics1_full_np = np.asarray(adata_omics1.obsm["feat"], dtype=np.float32)
    omics2_full_np = to_numpy_array(adata_omics2.X).astype(np.float32)
    return finalize_domain_graph(
        adata_omics1=adata_omics1,
        adata_omics2=adata_omics2,
        omics1_full_np=omics1_full_np,
        omics2_full_np=omics2_full_np,
        dataset_cfg=dataset_cfg,
        random_seed=random_seed,
        use_niche_graph=use_niche_graph,
        niche_graph_weight=niche_graph_weight,
        niche_feature_mode=niche_feature_mode,
        full_pair_niche_basis=full_pair_niche_basis,
        niche_graph_neighbors=niche_graph_neighbors,
        niche_graph_max_cosine_distance=niche_graph_max_cosine_distance,
        feature_representation={
            "omics1": {
                "modality": "RNA",
                "preprocessing": "filter_min_cells_hvg_seurat_v3_normalize_total_1e4_log1p_scale_pca",
                "requested_pca_dim": 20,
                "model_input_dim": int(omics1_full_np.shape[1]),
            },
            "omics2": {
                "modality": dataset_cfg.modality_kind,
                "preprocessing": "clr_scale",
                "model_input_dim": int(omics2_full_np.shape[1]),
            },
        },
    )



def preprocess_spatialglue_unlabeled(
    adata_omics1,
    adata_omics2,
    dataset_cfg: DatasetDefaults,
    random_seed: int,
    use_niche_graph: bool,
    niche_graph_weight: float,
    niche_feature_mode: str = "mean_std",
    full_pair_niche_basis: str = "joint",
    niche_graph_neighbors: int | None = None,
    niche_graph_max_cosine_distance: float | None = None,
    rna_min_cells: int = 10,
    rna_hvg_top_genes: int = 3000,
    rna_pca_dim: int = 50,
    prefer_existing_rna_pca: bool = True,
    omics2_hvg_top_features: int = 3000,
    omics2_pca_dim: int = 50,
    prefer_existing_omics2_lsi: bool = True,
    omics2_use_clr: bool = True,
    omics2_scale: bool = True,
):
    prefer_existing_rna_pca = parse_bool_option(prefer_existing_rna_pca, default=True)
    prefer_existing_omics2_lsi = parse_bool_option(prefer_existing_omics2_lsi, default=True)
    omics2_use_clr = parse_bool_option(omics2_use_clr, default=True)
    omics2_scale = parse_bool_option(omics2_scale, default=True)

    if prefer_existing_rna_pca and "X_pca" in adata_omics1.obsm:
        requested_rna_dim = int(rna_pca_dim)
        rna_dim = safe_n_comps(requested_rna_dim, adata_omics1.obsm["X_pca"].shape[1], adata_omics1.n_obs - 1)
        adata_omics1.obsm["feat"] = np.asarray(adata_omics1.obsm["X_pca"][:, :rna_dim], dtype=np.float32)
        rna_representation = {
            "modality": "RNA",
            "preprocessing": "use_existing_X_pca",
            "use_existing_obsm": "X_pca",
            "requested_pca_dim": requested_rna_dim,
            "model_input_dim": int(rna_dim),
            "selected_feature_dim": int(adata_omics1.n_vars),
        }
    else:
        filter_genes_min_cells_fast(adata_omics1, min_cells=int(rna_min_cells))
        effective_hvg = min(int(rna_hvg_top_genes), int(adata_omics1.n_vars))
        if effective_hvg > 0 and effective_hvg < int(adata_omics1.n_vars):
            sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=effective_hvg)
            rna_hvg_mask = adata_omics1.var["highly_variable"].to_numpy()
        else:
            adata_omics1.var["highly_variable"] = True
            rna_hvg_mask = np.ones(adata_omics1.n_vars, dtype=bool)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        sc.pp.scale(adata_omics1)
        adata_omics1 = adata_omics1[:, rna_hvg_mask].copy()
        rna_dim = safe_n_comps(rna_pca_dim, adata_omics1.n_obs - 1, adata_omics1.n_vars - 1)
        adata_omics1.obsm["feat"] = pca(adata_omics1, n_comps=rna_dim).astype(np.float32, copy=False)
        rna_representation = {
            "modality": "RNA",
            "preprocessing": "filter_min_cells_hvg_seurat_v3_normalize_total_1e4_log1p_scale_pca",
            "min_cells": int(rna_min_cells),
            "requested_hvg_top_genes": int(rna_hvg_top_genes),
            "selected_feature_dim": int(adata_omics1.n_vars),
            "requested_pca_dim": int(rna_pca_dim),
            "model_input_dim": int(rna_dim),
        }

    modality_upper = dataset_cfg.modality_kind.upper()
    if modality_upper == "ATAC":
        if prefer_existing_omics2_lsi and "X_lsi" in adata_omics2.obsm:
            requested_dim = int(omics2_pca_dim)
            omics2_dim = safe_n_comps(requested_dim, adata_omics2.obsm["X_lsi"].shape[1], adata_omics2.n_obs - 1)
            adata_omics2.obsm["feat"] = np.asarray(adata_omics2.obsm["X_lsi"][:, :omics2_dim], dtype=np.float32)
            omics2_representation = {
                "modality": dataset_cfg.modality_kind,
                "preprocessing": "use_existing_X_lsi",
                "use_existing_obsm": "X_lsi",
                "requested_dim": requested_dim,
                "model_input_dim": int(omics2_dim),
                "selected_feature_dim": int(adata_omics2.n_vars),
            }
        else:
            effective_hvg = min(int(omics2_hvg_top_features), int(adata_omics2.n_vars))
            if effective_hvg > 0 and effective_hvg < int(adata_omics2.n_vars):
                sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=effective_hvg)
                adata_omics2 = adata_omics2[:, adata_omics2.var["highly_variable"]].copy()
            lsi_dim = safe_n_comps(omics2_pca_dim + 1, adata_omics2.n_obs - 1, adata_omics2.n_vars - 1)
            lsi(adata_omics2, use_highly_variable=False, n_components=lsi_dim)
            lsi_rep = np.asarray(adata_omics2.obsm["X_lsi"], dtype=np.float32)
            adata_omics2.obsm["feat"] = lsi_rep[:, 1:] if lsi_rep.shape[1] > 1 else lsi_rep
            omics2_representation = {
                "modality": dataset_cfg.modality_kind,
                "preprocessing": "hvg_seurat_v3_lsi_drop_first_component",
                "requested_hvg_top_features": int(omics2_hvg_top_features),
                "requested_lsi_dim": int(omics2_pca_dim),
                "model_input_dim": int(adata_omics2.obsm["feat"].shape[1]),
                "selected_feature_dim": int(adata_omics2.n_vars),
            }
    else:
        if omics2_use_clr:
            adata_omics2 = clr_normalize_each_cell(adata_omics2)
        if omics2_scale:
            sc.pp.scale(adata_omics2)
        if int(omics2_pca_dim) > 0 and int(adata_omics2.n_vars) > int(omics2_pca_dim):
            omics2_dim = safe_n_comps(omics2_pca_dim, adata_omics2.n_obs - 1, adata_omics2.n_vars - 1)
            adata_omics2.obsm["feat"] = pca(adata_omics2, n_comps=omics2_dim).astype(np.float32, copy=False)
            preprocessing = "clr_scale_pca" if omics2_use_clr and omics2_scale else "custom_pca"
        else:
            adata_omics2.obsm["feat"] = to_numpy_array(adata_omics2.X).astype(np.float32)
            preprocessing = "clr_scale" if omics2_use_clr and omics2_scale else "custom"
        omics2_representation = {
            "modality": dataset_cfg.modality_kind,
            "preprocessing": preprocessing,
            "use_clr": int(omics2_use_clr),
            "scale": int(omics2_scale),
            "requested_pca_dim": int(omics2_pca_dim),
            "model_input_dim": int(adata_omics2.obsm["feat"].shape[1]),
            "selected_feature_dim": int(adata_omics2.n_vars),
        }

    omics1_full_np = np.asarray(adata_omics1.obsm["feat"], dtype=np.float32)
    omics2_full_np = np.asarray(adata_omics2.obsm["feat"], dtype=np.float32)
    return finalize_domain_graph(
        adata_omics1=adata_omics1,
        adata_omics2=adata_omics2,
        omics1_full_np=omics1_full_np,
        omics2_full_np=omics2_full_np,
        dataset_cfg=dataset_cfg,
        random_seed=random_seed,
        use_niche_graph=use_niche_graph,
        niche_graph_weight=niche_graph_weight,
        niche_feature_mode=niche_feature_mode,
        full_pair_niche_basis=full_pair_niche_basis,
        niche_graph_neighbors=niche_graph_neighbors,
        niche_graph_max_cosine_distance=niche_graph_max_cosine_distance,
        feature_representation={
            "omics1": rna_representation,
            "omics2": omics2_representation,
        },
    )


def preprocess_e15(
    adata_omics1,
    adata_omics2,
    dataset_cfg: DatasetDefaults,
    random_seed: int,
    use_niche_graph: bool,
    niche_graph_weight: float,
    niche_feature_mode: str = "mean_std",
    full_pair_niche_basis: str = "joint",
    niche_graph_neighbors: int | None = None,
    niche_graph_max_cosine_distance: float | None = None,
    rna_use_pca: bool = True,
    rna_pca_dim: int = 200,
):
    filter_genes_min_cells_fast(adata_omics1, min_cells=10)
    sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
    rna_hvg_mask = adata_omics1.var["highly_variable"].to_numpy()
    sc.pp.normalize_total(adata_omics1, target_sum=1e4)
    sc.pp.log1p(adata_omics1)
    sc.pp.scale(adata_omics1)
    adata_omics1 = adata_omics1[:, rna_hvg_mask].copy()
    rna_use_pca = bool(rna_use_pca)
    rna_preprocessed_np = to_numpy_array(adata_omics1.X).astype(np.float32, copy=False)

    rna_model_dim = safe_n_comps(rna_pca_dim, adata_omics1.n_obs - 1, adata_omics1.n_vars - 1)
    adata_omics1.obsm["feat"] = pca(adata_omics1, n_comps=rna_model_dim).astype(np.float32, copy=False)
    rna_representation = {
        "modality": "RNA",
        "preprocessing": "normalize_total_1e4_log1p_scale",
        "use_pca": 1,
        "requested_pca_dim": int(rna_pca_dim),
        "model_input_dim": int(rna_model_dim),
        "preprocessed_dim": int(rna_preprocessed_np.shape[1]),
    }

    sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)
    adata_omics2 = adata_omics2[:, adata_omics2.var["highly_variable"]].copy()
    lsi(adata_omics2, use_highly_variable=False, n_components=101)
    adata_omics2.obsm["feat"] = adata_omics2.obsm["X_lsi"].copy()

    omics1_full_np = np.asarray(adata_omics1.obsm["feat"], dtype=np.float32)
    omics2_full_np = np.asarray(adata_omics2.obsm["feat"], dtype=np.float32)
    return finalize_domain_graph(
        adata_omics1=adata_omics1,
        adata_omics2=adata_omics2,
        omics1_full_np=omics1_full_np,
        omics2_full_np=omics2_full_np,
        dataset_cfg=dataset_cfg,
        random_seed=random_seed,
        use_niche_graph=use_niche_graph,
        niche_graph_weight=niche_graph_weight,
        niche_feature_mode=niche_feature_mode,
        full_pair_niche_basis=full_pair_niche_basis,
        niche_graph_neighbors=niche_graph_neighbors,
        niche_graph_max_cosine_distance=niche_graph_max_cosine_distance,
        feature_representation={
            "omics1": rna_representation,
            "omics2": {
                "modality": dataset_cfg.modality_kind,
                "preprocessing": "hvg_seurat_v3_3000_lsi_101_drop_first_component",
                "model_input_dim": int(omics2_full_np.shape[1]),
                "selected_feature_dim": int(adata_omics2.n_vars),
            },
        },
    )


def finalize_domain_graph(
    *,
    adata_omics1,
    adata_omics2,
    omics1_full_np: np.ndarray,
    omics2_full_np: np.ndarray,
    dataset_cfg: DatasetDefaults,
    random_seed: int,
    use_niche_graph: bool,
    niche_graph_weight: float,
    niche_feature_mode: str = "mean_std",
    full_pair_niche_basis: str = "joint",
    niche_graph_neighbors: int | None = None,
    niche_graph_max_cosine_distance: float | None = None,
    feature_representation: dict | None = None,
):
    spatial_coords = np.asarray(adata_omics1.obsm["spatial"], dtype=np.float32)
    full_pair_niche_basis = str(full_pair_niche_basis or "joint").lower()
    if full_pair_niche_basis == "omics1":
        niche_basis = omics1_full_np
    elif full_pair_niche_basis == "omics2":
        niche_basis = omics2_full_np
    elif full_pair_niche_basis == "joint":
        niche_basis = np.concatenate([omics1_full_np, omics2_full_np], axis=1)
    else:
        raise ValueError(f"Unsupported full_pair_niche_basis: {full_pair_niche_basis}")

    spatial_adj_dense = create_adj_dense(adata_omics1, n_neighbors=dataset_cfg.spatial_neighbors)
    niche_features_np = build_niche_features(
        niche_basis,
        spatial_coords,
        n_neighbors=dataset_cfg.niche_neighbors,
        include_self=True,
        add_std=True,
        covariance_mode=niche_feature_mode,
        random_seed=random_seed,
    )
    niche_adj_dense = None
    adj_mixed_dense = spatial_adj_dense.copy()
    effective_niche_graph_neighbors = (
        int(dataset_cfg.niche_graph_neighbors) if niche_graph_neighbors is None else int(niche_graph_neighbors)
    )
    effective_niche_graph_max_cosine_distance = (
        dataset_cfg.niche_graph_max_cosine_distance
        if niche_graph_max_cosine_distance is None
        else float(niche_graph_max_cosine_distance)
    )
    if use_niche_graph:
        niche_adj_dense = construct_niche_graph(
            niche_features_np,
            n_neighbors=effective_niche_graph_neighbors,
            metric="cosine",
            return_dense=True,
            max_distance=effective_niche_graph_max_cosine_distance,
        )
        adj_mixed_dense = blend_adjacency_matrices(spatial_adj_dense, niche_adj_dense, niche_weight=niche_graph_weight)

    print(f"RNA feat for model: {omics1_full_np.shape}")
    print(f"{dataset_cfg.modality_kind} feat for model: {omics2_full_np.shape}")
    print(f"Use niche graph: {use_niche_graph}")
    print(f"Niche feature mode: {niche_feature_mode}")
    print(f"Niche graph neighbors: {effective_niche_graph_neighbors}")
    print(f"Niche graph max cosine distance: {effective_niche_graph_max_cosine_distance}")
    print("Niche feature shape:", niche_features_np.shape)
    print("Niche adj shape:", None if niche_adj_dense is None else niche_adj_dense.shape)

    return {
        "adata_omics1": adata_omics1,
        "adata_omics2": adata_omics2,
        "spatial_coords": spatial_coords,
        "omics1_full_np": omics1_full_np,
        "omics2_full_np": omics2_full_np,
        "input_omics1_np": omics1_full_np.copy(),
        "input_omics2_np": omics2_full_np.copy(),
        "niche_features_np": niche_features_np,
        "niche_feature_mode": niche_feature_mode,
        "niche_graph_neighbors": effective_niche_graph_neighbors,
        "niche_graph_max_cosine_distance": effective_niche_graph_max_cosine_distance,
        "feature_representation": feature_representation or {},
        "spatial_adj_dense": spatial_adj_dense,
        "niche_adj_dense_static": niche_adj_dense,
        "adj_mixed_dense_static": adj_mixed_dense,
    }


def preprocess_dataset(
    dataset_cfg: DatasetDefaults,
    adata_omics1,
    adata_omics2,
    random_seed: int,
    use_niche_graph: bool,
    niche_graph_weight: float,
    niche_feature_mode: str = "mean_std",
    full_pair_niche_basis: str = "joint",
    niche_graph_neighbors: int | None = None,
    niche_graph_max_cosine_distance: float | None = None,
    e15_rna_use_pca: bool = True,
    e15_rna_pca_dim: int = 100,
    spatialglue_rna_min_cells: int = 10,
    spatialglue_rna_hvg_top_genes: int = 3000,
    spatialglue_rna_pca_dim: int = 50,
    spatialglue_prefer_existing_rna_pca: bool = True,
    spatialglue_omics2_hvg_top_features: int = 3000,
    spatialglue_omics2_pca_dim: int = 50,
    spatialglue_prefer_existing_omics2_lsi: bool = True,
    spatialglue_omics2_use_clr: bool = True,
    spatialglue_omics2_scale: bool = True,
):
    dataset_name = str(dataset_cfg.dataset_name).lower()
    if dataset_name == "p22" or dataset_name.startswith("spatialglue_mouse_") or dataset_name.startswith("mouse_"):
        return preprocess_spatialglue_unlabeled(
            adata_omics1,
            adata_omics2,
            dataset_cfg,
            random_seed,
            use_niche_graph,
            niche_graph_weight,
            niche_feature_mode,
            full_pair_niche_basis,
            niche_graph_neighbors,
            niche_graph_max_cosine_distance,
            spatialglue_rna_min_cells,
            spatialglue_rna_hvg_top_genes,
            spatialglue_rna_pca_dim,
            spatialglue_prefer_existing_rna_pca,
            spatialglue_omics2_hvg_top_features,
            spatialglue_omics2_pca_dim,
            spatialglue_prefer_existing_omics2_lsi,
            spatialglue_omics2_use_clr,
            spatialglue_omics2_scale,
        )
    if dataset_name in {"e15", "e15_domain_only"}:
        return preprocess_e15(
            adata_omics1,
            adata_omics2,
            dataset_cfg,
            random_seed,
            use_niche_graph,
            niche_graph_weight,
            niche_feature_mode,
            full_pair_niche_basis,
            niche_graph_neighbors,
            niche_graph_max_cosine_distance,
            e15_rna_use_pca,
            e15_rna_pca_dim,
        )
    if dataset_name == "a1" or dataset_cfg.modality_kind.upper() == "ADT":
        return preprocess_a1(
            adata_omics1,
            adata_omics2,
            dataset_cfg,
            random_seed,
            use_niche_graph,
            niche_graph_weight,
            niche_feature_mode,
            full_pair_niche_basis,
            niche_graph_neighbors,
            niche_graph_max_cosine_distance,
        )
    raise ValueError(f"Unsupported dataset: {dataset_cfg.dataset_name}")


def hungarian_align_labels(y_true, y_pred):
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    true_classes = np.unique(y_true)
    pred_classes = np.unique(y_pred)
    cost = np.zeros((len(true_classes), len(pred_classes)), dtype=np.int64)
    for i, true_label in enumerate(true_classes):
        true_mask = y_true == true_label
        for j, pred_label in enumerate(pred_classes):
            pred_mask = y_pred == pred_label
            cost[i, j] = -np.sum(true_mask & pred_mask)
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {pred_classes[col]: true_classes[row] for row, col in zip(row_ind, col_ind)}
    aligned = np.array([mapping.get(label, label) for label in y_pred], dtype=object)
    return aligned, mapping


def plot_outputs(*, adata_eval, labels_true: np.ndarray, pred_aligned: np.ndarray, run_tag: str, figure_dir: Path, dataset_cfg: DatasetDefaults, save_outputs: bool) -> None:
    has_umap = "X_umap" in adata_eval.obsm
    ncols = 2 if has_umap else 1
    fig, axes = plt.subplots(1, ncols, figsize=(14, 6) if has_umap else (7, 6))
    axes = np.atleast_1d(axes)
    spatial_ax = axes[1] if has_umap else axes[0]
    if has_umap:
        sc.pl.umap(adata_eval, color="pred_aligned", title="UMAP (aligned pred)", ax=axes[0], legend_loc=None, show=False, size=40)
    sc.pl.embedding(
        adata_eval,
        basis="spatial",
        color="pred_aligned",
        title=run_tag,
        ax=spatial_ax,
        show=False,
        size=dataset_cfg.pair_plot_size,
    )
    plt.tight_layout()
    if save_outputs:
        save_figure(fig, figure_dir / f"spatial_prediction_{run_tag}.png")
    else:
        plt.close(fig)

    conf_mat = pd.crosstab(pd.Series(labels_true, name="label"), pd.Series(pred_aligned, name="pred_aligned"))
    conf_mat_norm = conf_mat.div(conf_mat.sum(axis=1), axis=0).fillna(0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(conf_mat.values, cmap="Blues")
    axes[0].set_title("Confusion matrix")
    axes[0].set_xticks(np.arange(len(conf_mat.columns)))
    axes[0].set_yticks(np.arange(len(conf_mat.index)))
    axes[0].set_xticklabels(conf_mat.columns, rotation=60, ha="right")
    axes[0].set_yticklabels(conf_mat.index)
    axes[1].imshow(conf_mat_norm.values, cmap="OrRd", vmin=0, vmax=1)
    axes[1].set_title("Row-normalized confusion matrix")
    axes[1].set_xticks(np.arange(len(conf_mat_norm.columns)))
    axes[1].set_yticks(np.arange(len(conf_mat_norm.index)))
    axes[1].set_xticklabels(conf_mat_norm.columns, rotation=60, ha="right")
    axes[1].set_yticklabels(conf_mat_norm.index)
    plt.tight_layout()
    if save_outputs:
        save_figure(fig, figure_dir / f"confusion_matrix_{run_tag}.png")
    else:
        plt.close(fig)


def plot_unlabeled_outputs(*, adata_eval, run_tag: str, figure_dir: Path, dataset_cfg: DatasetDefaults, save_outputs: bool) -> None:
    has_umap = "X_umap" in adata_eval.obsm
    ncols = 2 if has_umap else 1
    fig, axes = plt.subplots(1, ncols, figsize=(14, 6) if has_umap else (7, 6))
    axes = np.atleast_1d(axes)
    spatial_ax = axes[1] if has_umap else axes[0]
    if has_umap:
        sc.pl.umap(adata_eval, color="pred", title="UMAP (pred)", ax=axes[0], legend_loc=None, show=False, size=40)
    sc.pl.embedding(
        adata_eval,
        basis="spatial",
        color="pred",
        title=run_tag,
        ax=spatial_ax,
        show=False,
        size=dataset_cfg.pair_plot_size,
    )
    plt.tight_layout()
    if save_outputs:
        save_figure(fig, figure_dir / f"spatial_prediction_{run_tag}.png")
    else:
        plt.close(fig)


def run_single_experiment(
    dataset_cfg: DatasetDefaults,
    preset_name: str,
    preset: dict,
    random_seed: int,
    results_root: Path,
    device: torch.device,
    save_outputs: bool = True,
    save_figures: bool = True,
):
    print("=" * 80)
    print(f"Dataset: {dataset_cfg.dataset_name} | Preset: {preset_name} | Seed: {random_seed}")
    print(f"Device: {device}")

    run_tag, dataset_result_dir, figure_dir = build_run_outputs(dataset_cfg.dataset_name, preset_name, random_seed, results_root)
    adata_omics1, adata_omics2, labels_true = load_dataset(dataset_cfg)
    model_name = preset.get("model_name", "spanichepair_gcnii")
    loss_n = float(preset.get("loss_n", 0.0))
    loss_w = float(preset.get("loss_w", dataset_cfg.loss_w))
    loss_s = float(preset.get("loss_s", dataset_cfg.loss_s))
    early_stop_unsupervised = int(preset.get("early_stop_unsupervised", 0))
    early_stop_tool = str(preset.get("early_stop_tool", dataset_cfg.primary_tool or "kmeans"))
    early_stop_metric = str(preset.get("early_stop_metric", "silhouette"))
    early_stop_warmup_ratio = float(preset.get("early_stop_warmup_ratio", 0.25))
    early_stop_patience = int(preset.get("early_stop_patience", 8))
    early_stop_min_delta = float(preset.get("early_stop_min_delta", 1e-4))
    early_stop_sample_size = int(preset.get("early_stop_sample_size", 2000))
    train_eval_interval = int(preset.get("train_eval_interval", 10))
    niche_graph_weight = float(preset["niche_graph_weight"])
    use_niche_graph = bool(preset.get("use_niche_graph", True))
    full_pair_niche_basis = str(preset.get("full_pair_niche_basis", "joint"))
    niche_feature_mode = str(preset.get("niche_feature_mode", "mean_std"))
    niche_graph_neighbors = int(preset.get("niche_graph_neighbors", dataset_cfg.niche_graph_neighbors))
    niche_graph_max_cosine_distance = preset.get(
        "niche_graph_max_cosine_distance",
        dataset_cfg.niche_graph_max_cosine_distance,
    )
    if niche_graph_max_cosine_distance is not None:
        niche_graph_max_cosine_distance = float(niche_graph_max_cosine_distance)
    use_dynamic_niche_graph = bool(preset.get("use_dynamic_niche_graph", False))
    dynamic_graph_warmup = float(preset.get("dynamic_graph_warmup", 0.2))
    dynamic_graph_update_interval = int(preset.get("dynamic_graph_update_interval", 50))
    dynamic_graph_smooth_beta = float(preset.get("dynamic_graph_smooth_beta", 0.2))
    dynamic_graph_alpha_min = float(preset.get("dynamic_graph_alpha_min", 0.05))
    dynamic_graph_adaptive_alpha = int(preset.get("dynamic_graph_adaptive_alpha", 0))
    dynamic_graph_beta_min = float(preset.get("dynamic_graph_beta_min", 0.02))
    dynamic_graph_adaptive_beta = int(preset.get("dynamic_graph_adaptive_beta", 1))
    dynamic_graph_metric = str(preset.get("dynamic_graph_metric", "cosine"))
    dynamic_graph_source = str(preset.get("dynamic_graph_source", "joint"))
    gcnii_alpha = float(preset.get("gcnii_alpha", 0.3))
    gcnii_lambda = float(preset.get("gcnii_lambda", 0.5))
    n_z = int(preset.get("n_z", 40))
    graph_hidden_dim = int(preset.get("graph_hidden_dim", 64))
    gcnii_num_layers = int(preset.get("gcnii_num_layers", 4))
    use_boundary_aware_gcn = int(preset.get("use_boundary_aware_gcn", 0))
    boundary_gate_temperature = float(preset.get("boundary_gate_temperature", 0.5))
    boundary_gate_min = float(preset.get("boundary_gate_min", 0.2))
    e15_rna_use_pca = parse_bool_option(preset.get("e15_rna_use_pca", True), default=True)
    e15_rna_pca_dim = int(preset.get("e15_rna_pca_dim", 200))
    processed = preprocess_dataset(
        dataset_cfg=dataset_cfg,
        adata_omics1=adata_omics1,
        adata_omics2=adata_omics2,
        random_seed=random_seed,
        use_niche_graph=use_niche_graph,
        niche_graph_weight=niche_graph_weight,
        niche_feature_mode=niche_feature_mode,
        full_pair_niche_basis=full_pair_niche_basis,
        niche_graph_neighbors=niche_graph_neighbors,
        niche_graph_max_cosine_distance=niche_graph_max_cosine_distance,
        e15_rna_use_pca=e15_rna_use_pca,
        e15_rna_pca_dim=e15_rna_pca_dim,
    )

    classes, label = np.unique(labels_true, return_inverse=True)
    args = build_args(
        dataset_name=dataset_cfg.dataset_name,
        n_input=processed["omics1_full_np"].shape[1],
        n_input1=processed["omics2_full_np"].shape[1],
        n_clusters=len(classes),
        random_seed=random_seed,
        epoch=dataset_cfg.epoch,
        lr=dataset_cfg.lr,
        sigma=dataset_cfg.sigma,
        loss_n=loss_n,
        loss_w=loss_w,
        loss_s=loss_s,
        loss_a=dataset_cfg.loss_a,
        early_stop_unsupervised=early_stop_unsupervised,
        early_stop_tool=early_stop_tool,
        early_stop_metric=early_stop_metric,
        early_stop_warmup_ratio=early_stop_warmup_ratio,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
        early_stop_sample_size=early_stop_sample_size,
        train_eval_interval=train_eval_interval,
        contrastive_start_ratio=dataset_cfg.contrastive_start_ratio,
        contrastive_ramp_ratio=dataset_cfg.contrastive_ramp_ratio,
        contrastive_min_scale=dataset_cfg.contrastive_min_scale,
        contrastive_max_scale=dataset_cfg.contrastive_max_scale,
        contrastive_alignment_power=dataset_cfg.contrastive_alignment_power,
        contrastive_balance_floor=dataset_cfg.contrastive_balance_floor,
        n_z=n_z,
        model_name=model_name,
        gcnii_alpha=gcnii_alpha,
        gcnii_lambda=gcnii_lambda,
        graph_hidden_dim=graph_hidden_dim,
        gcnii_num_layers=gcnii_num_layers,
        use_boundary_aware_gcn=use_boundary_aware_gcn,
        boundary_gate_temperature=boundary_gate_temperature,
        boundary_gate_min=boundary_gate_min,
    )
    args.fusion_mode = dataset_cfg.fusion_mode
    args.fusion_hidden_dim = dataset_cfg.fusion_hidden_dim
    args.use_dynamic_niche_graph = int(use_dynamic_niche_graph)
    args.dynamic_graph_warmup = dynamic_graph_warmup
    args.dynamic_graph_update_interval = dynamic_graph_update_interval
    args.dynamic_graph_smooth_beta = dynamic_graph_smooth_beta
    args.dynamic_graph_alpha_min = dynamic_graph_alpha_min
    args.dynamic_graph_adaptive_alpha = dynamic_graph_adaptive_alpha
    args.dynamic_graph_beta_min = dynamic_graph_beta_min
    args.dynamic_graph_adaptive_beta = dynamic_graph_adaptive_beta
    args.dynamic_graph_metric = dynamic_graph_metric
    args.dynamic_graph_source = dynamic_graph_source
    args.dynamic_graph_neighbors = int(niche_graph_neighbors)
    args.dynamic_graph_niche_weight = float(niche_graph_weight)

    fix_seed(random_seed)
    adj_train = preprocess_graph(processed["adj_mixed_dense_static"]).to(device)
    features_omics1 = torch.FloatTensor(processed["input_omics1_np"]).to(device)
    features_omics2 = torch.FloatTensor(processed["input_omics2_np"]).to(device)
    full_features_omics1 = torch.FloatTensor(processed["omics1_full_np"]).to(device)
    full_features_omics2 = torch.FloatTensor(processed["omics2_full_np"]).to(device)
    if features_omics1.shape[1] != int(args.n_input):
        raise ValueError(f"omics1 feature dim {features_omics1.shape[1]} does not match args.n_input {args.n_input}.")
    if features_omics2.shape[1] != int(args.n_input1):
        raise ValueError(f"omics2 feature dim {features_omics2.shape[1]} does not match args.n_input1 {args.n_input1}.")

    model = create_model(model_name, args).to(device)
    model = Train(
        args.epoch,
        model,
        features_omics1,
        features_omics2,
        adj_train,
        label,
        device,
        args,
        data_target=full_features_omics1,
        data_target1=full_features_omics2,
        adj_spatial=processed["spatial_adj_dense"],
        adj_niche=processed["niche_adj_dense_static"],
        dynamic_spatial_adj_dense=processed["spatial_adj_dense"],
        dynamic_niche_static_adj_dense=processed["niche_adj_dense_static"],
        dynamic_mixed_static_adj_dense=processed["adj_mixed_dense_static"],
    )

    dynamic_graph_state = getattr(model, "last_dynamic_graph", None) or {}
    final_mixed_adj_dense = dynamic_graph_state.get("mixed_adj_dense")
    adj_test = preprocess_graph(final_mixed_adj_dense).to(device) if final_mixed_adj_dense is not None else adj_train
    dynamic_graph_summary = {key: to_builtin(value) for key, value in dynamic_graph_state.items() if key not in {"mixed_adj_dense", "niche_adj_dense"}}

    primary_tool = dataset_cfg.primary_tool
    nmi, ari, ami, homogeneity, completeness, v_measure = Test(
        model,
        processed["adata_omics1"],
        features_omics1,
        features_omics2,
        adj_test,
        label,
        device,
        args,
        primary_tool,
        data_target=full_features_omics1,
        data_target1=full_features_omics2,
        adj_spatial=processed["spatial_adj_dense"],
    )
    primary_last_test = copy.deepcopy(model.last_test)
    print(
        f"Primary ({primary_tool}) -> NMI={nmi:.4f}, ARI={ari:.4f}, "
        f"AMI={ami:.4f}, Homogeneity={homogeneity:.4f}, Completeness={completeness:.4f}, V-measure={v_measure:.4f}"
    )

    secondary_metrics = None
    if dataset_cfg.secondary_tool:
        nmi2, ari2, ami2, homogeneity2, completeness2, v_measure2 = Test(
            model,
            processed["adata_omics1"],
            features_omics1,
            features_omics2,
            adj_test,
            label,
            device,
            args,
            dataset_cfg.secondary_tool,
            data_target=full_features_omics1,
            data_target1=full_features_omics2,
            adj_spatial=processed["spatial_adj_dense"],
        )
        model.last_test = copy.deepcopy(primary_last_test)
        secondary_metrics = {
            "tool": dataset_cfg.secondary_tool,
            "nmi": float(nmi2),
            "ari": float(ari2),
            "ami": float(ami2),
            "homogeneity": float(homogeneity2),
            "completeness": float(completeness2),
            "v_measure": float(v_measure2),
        }
        print(
            f"Secondary ({dataset_cfg.secondary_tool}) -> NMI={nmi2:.4f}, ARI={ari2:.4f}, "
            f"AMI={ami2:.4f}, Homogeneity={homogeneity2:.4f}, Completeness={completeness2:.4f}, V-measure={v_measure2:.4f}"
        )

    embedding = np.asarray(model.last_test["embedding"], dtype=np.float32)
    pred = np.asarray(model.last_test["pred"]).astype(str)
    pred_aligned, _ = hungarian_align_labels(labels_true, pred)

    adata_eval = processed["adata_omics1"].copy()
    adata_eval.obsm["X_spaNicheGraph"] = embedding
    adata_eval.obs["label"] = pd.Categorical(labels_true.astype(str))
    adata_eval.obs["pred"] = pd.Categorical(pred.astype(str))
    adata_eval.obs["pred_aligned"] = pd.Categorical(pred_aligned.astype(str))
    try:
        sc.pp.neighbors(adata_eval, use_rep="X_spaNicheGraph", n_neighbors=30)
        sc.tl.umap(adata_eval, random_state=random_seed)
    except Exception as exc:  # UMAP is a convenience output, not a training requirement.
        print(f"Skip UMAP for {run_tag}: {type(exc).__name__}: {exc}")

    primary_metrics = {
        "tool": primary_tool,
        "nmi": float(nmi),
        "ari": float(ari),
        "ami": float(ami),
        "homogeneity": float(homogeneity),
        "completeness": float(completeness),
        "v_measure": float(v_measure),
    }
    metrics_payload = {
        "dataset": dataset_cfg.dataset_name,
        "experiment_preset": preset_name,
        "random_seed": int(random_seed),
        "model_name": model_name,
        "full_pair_niche_basis": full_pair_niche_basis,
        "niche_graph_neighbors": int(niche_graph_neighbors),
        "niche_graph_max_cosine_distance": (
            None if niche_graph_max_cosine_distance is None else float(niche_graph_max_cosine_distance)
        ),
        "loss_n": float(loss_n),
        "loss_w": float(loss_w),
        "loss_s": float(loss_s),
        "loss_a": float(dataset_cfg.loss_a),
        "early_stop_unsupervised": int(early_stop_unsupervised),
        "early_stop_tool": early_stop_tool,
        "early_stop_metric": early_stop_metric,
        "early_stop_warmup_ratio": float(early_stop_warmup_ratio),
        "early_stop_patience": int(early_stop_patience),
        "early_stop_min_delta": float(early_stop_min_delta),
        "early_stop_sample_size": int(early_stop_sample_size),
        "train_eval_interval": int(train_eval_interval),
        "early_stop": to_builtin(getattr(model, "early_stop_state", {}) or {}),
        "n_z": int(n_z),
        "model_input": {
            "omics1_dim": int(args.n_input),
            "omics2_dim": int(args.n_input1),
        },
        "feature_representation": processed.get("feature_representation") or {},
        "graph_hidden_dim": int(graph_hidden_dim),
        "gcnii_num_layers": int(gcnii_num_layers),
        "gcnii_alpha": float(gcnii_alpha),
        "gcnii_lambda": float(gcnii_lambda),
        "use_boundary_aware_gcn": int(use_boundary_aware_gcn),
        "boundary_gate_temperature": float(boundary_gate_temperature),
        "boundary_gate_min": float(boundary_gate_min),
        "boundary_gate": to_builtin(model.last_test.get("boundary_gate") or {}),
        "primary": primary_metrics,
        "secondary": secondary_metrics,
        "branch_state": to_builtin(model.last_test.get("branch_state") or {}),
        "niche_feature": {
            "mode": processed.get("niche_feature_mode"),
            "feature_dim": None if processed.get("niche_features_np") is None else int(processed["niche_features_np"].shape[1]),
            "niche_graph_neighbors": processed.get("niche_graph_neighbors"),
            "niche_graph_max_cosine_distance": processed.get("niche_graph_max_cosine_distance"),
        },
        "graph_structure": {
            "spatial": adjacency_edge_stats(processed.get("spatial_adj_dense")),
            "niche_static": adjacency_edge_stats(processed.get("niche_adj_dense_static")),
            "mixed_static": adjacency_edge_stats(processed.get("adj_mixed_dense_static")),
            "niche_final": adjacency_edge_stats(dynamic_graph_state.get("niche_adj_dense")),
            "mixed_final": adjacency_edge_stats(dynamic_graph_state.get("mixed_adj_dense")),
        },
        "dynamic_graph": dynamic_graph_summary,
    }

    spatial_coords = np.asarray(adata_eval.obsm["spatial"], dtype=np.float32)
    pred_df = pd.DataFrame(
        {
            "spot_id": adata_eval.obs_names.astype(str),
            "label": labels_true.astype(str),
            "pred": pred.astype(str),
            "pred_aligned": pred_aligned.astype(str),
            "spatial1": spatial_coords[:, 0],
            "spatial2": spatial_coords[:, 1],
        }
    )

    if save_outputs:
        save_metrics_bundle(metrics_payload, dataset_result_dir, run_tag)
        pred_df.to_csv(dataset_result_dir / f"pred_labels_{run_tag}.csv", index=False, encoding="utf-8-sig")
        save_adata_safe(adata_eval, dataset_result_dir / f"adata_plot_{run_tag}.h5ad")
        if dynamic_graph_state.get("mixed_adj_dense") is not None:
            np.save(dataset_result_dir / f"final_mixed_adj_{run_tag}.npy", np.asarray(dynamic_graph_state["mixed_adj_dense"], dtype=np.float32))
        if dynamic_graph_state.get("niche_adj_dense") is not None:
            np.save(dataset_result_dir / f"final_niche_adj_{run_tag}.npy", np.asarray(dynamic_graph_state["niche_adj_dense"], dtype=np.float32))

    if save_figures:
        plot_outputs(
            adata_eval=adata_eval,
            labels_true=labels_true,
            pred_aligned=pred_aligned,
            run_tag=run_tag,
            figure_dir=figure_dir,
            dataset_cfg=dataset_cfg,
            save_outputs=save_outputs,
        )

    train_df = pd.DataFrame(getattr(model, "training_history", []))
    if not train_df.empty:
        if save_outputs:
            train_df.to_csv(dataset_result_dir / f"training_history_{run_tag}.csv", index=False, encoding="utf-8-sig")
        if save_figures:
            fig, ax = plt.subplots(1, 1, figsize=(7, 4))
            ax.plot(train_df["epoch"], train_df["loss_total"], label="loss_total")
            for key in ["loss_ae", "loss_w", "loss_igae", "loss_adj", "loss_s", "loss_nce", "early_stop_score"]:
                if key in train_df:
                    ax.plot(train_df["epoch"], train_df[key], label=key)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Value")
            ax.legend(fontsize=8)
            ax.set_title(f"Training curves: {run_tag}")
            plt.tight_layout()
            if save_outputs:
                save_figure(fig, figure_dir / f"training_curves_{run_tag}.png")
            else:
                plt.close(fig)

    return summarize_metrics_payload(metrics_payload)


def run_single_unlabeled_experiment(
    dataset_cfg: DatasetDefaults,
    preset_name: str,
    preset: dict,
    random_seed: int,
    results_root: Path,
    device: torch.device,
    save_outputs: bool = True,
    save_heavy_outputs: bool = True,
):
    print("=" * 80)
    print(f"Dataset: {dataset_cfg.dataset_name} | Preset: {preset_name} | Seed: {random_seed}")
    print(f"Device: {device}")
    print("Manual labels: unavailable; running label-free clustering and unsupervised diagnostics.")

    run_tag, dataset_result_dir, figure_dir = build_run_outputs(dataset_cfg.dataset_name, preset_name, random_seed, results_root)
    adata_omics1, adata_omics2 = load_unlabeled_dataset(dataset_cfg)
    model_name = preset.get("model_name", "spanichepair_gcnii")
    n_clusters = int(preset.get("n_clusters", preset.get("target_cluster_count", 8)))
    loss_n = float(preset.get("loss_n", 0.0))
    loss_w = float(preset.get("loss_w", dataset_cfg.loss_w))
    loss_s = float(preset.get("loss_s", dataset_cfg.loss_s))
    early_stop_unsupervised = int(preset.get("early_stop_unsupervised", 1))
    early_stop_tool = str(preset.get("early_stop_tool", dataset_cfg.primary_tool or "kmeans"))
    early_stop_metric = str(preset.get("early_stop_metric", "silhouette"))
    early_stop_warmup_ratio = float(preset.get("early_stop_warmup_ratio", 0.25))
    early_stop_patience = int(preset.get("early_stop_patience", 8))
    early_stop_min_delta = float(preset.get("early_stop_min_delta", 1e-4))
    early_stop_sample_size = int(preset.get("early_stop_sample_size", 2000))
    train_eval_interval = int(preset.get("train_eval_interval", 10))
    niche_graph_weight = float(preset["niche_graph_weight"])
    use_niche_graph = bool(preset.get("use_niche_graph", True))
    full_pair_niche_basis = str(preset.get("full_pair_niche_basis", "joint"))
    niche_feature_mode = str(preset.get("niche_feature_mode", "mean_std"))
    niche_graph_neighbors = int(preset.get("niche_graph_neighbors", dataset_cfg.niche_graph_neighbors))
    niche_graph_max_cosine_distance = preset.get(
        "niche_graph_max_cosine_distance",
        dataset_cfg.niche_graph_max_cosine_distance,
    )
    if niche_graph_max_cosine_distance is not None:
        niche_graph_max_cosine_distance = float(niche_graph_max_cosine_distance)
    use_dynamic_niche_graph = bool(preset.get("use_dynamic_niche_graph", False))
    dynamic_graph_warmup = float(preset.get("dynamic_graph_warmup", 0.2))
    dynamic_graph_update_interval = int(preset.get("dynamic_graph_update_interval", 50))
    dynamic_graph_smooth_beta = float(preset.get("dynamic_graph_smooth_beta", 0.2))
    dynamic_graph_alpha_min = float(preset.get("dynamic_graph_alpha_min", 0.05))
    dynamic_graph_adaptive_alpha = int(preset.get("dynamic_graph_adaptive_alpha", 0))
    dynamic_graph_beta_min = float(preset.get("dynamic_graph_beta_min", 0.02))
    dynamic_graph_adaptive_beta = int(preset.get("dynamic_graph_adaptive_beta", 1))
    dynamic_graph_metric = str(preset.get("dynamic_graph_metric", "cosine"))
    dynamic_graph_source = str(preset.get("dynamic_graph_source", "joint"))
    gcnii_alpha = float(preset.get("gcnii_alpha", 0.3))
    gcnii_lambda = float(preset.get("gcnii_lambda", 0.5))
    n_z = int(preset.get("n_z", 40))
    graph_hidden_dim = int(preset.get("graph_hidden_dim", 64))
    gcnii_num_layers = int(preset.get("gcnii_num_layers", 4))
    use_boundary_aware_gcn = int(preset.get("use_boundary_aware_gcn", 0))
    boundary_gate_temperature = float(preset.get("boundary_gate_temperature", 0.5))
    boundary_gate_min = float(preset.get("boundary_gate_min", 0.2))

    processed = preprocess_dataset(
        dataset_cfg=dataset_cfg,
        adata_omics1=adata_omics1,
        adata_omics2=adata_omics2,
        random_seed=random_seed,
        use_niche_graph=use_niche_graph,
        niche_graph_weight=niche_graph_weight,
        niche_feature_mode=niche_feature_mode,
        full_pair_niche_basis=full_pair_niche_basis,
        niche_graph_neighbors=niche_graph_neighbors,
        niche_graph_max_cosine_distance=niche_graph_max_cosine_distance,
        spatialglue_rna_min_cells=int(preset.get("spatialglue_rna_min_cells", 10)),
        spatialglue_rna_hvg_top_genes=int(preset.get("spatialglue_rna_hvg_top_genes", 3000)),
        spatialglue_rna_pca_dim=int(preset.get("spatialglue_rna_pca_dim", 50)),
        spatialglue_prefer_existing_rna_pca=parse_bool_option(
            preset.get("spatialglue_prefer_existing_rna_pca", True),
            default=True,
        ),
        spatialglue_omics2_hvg_top_features=int(preset.get("spatialglue_omics2_hvg_top_features", 3000)),
        spatialglue_omics2_pca_dim=int(preset.get("spatialglue_omics2_pca_dim", 50)),
        spatialglue_prefer_existing_omics2_lsi=parse_bool_option(
            preset.get("spatialglue_prefer_existing_omics2_lsi", True),
            default=True,
        ),
        spatialglue_omics2_use_clr=parse_bool_option(preset.get("spatialglue_omics2_use_clr", True), default=True),
        spatialglue_omics2_scale=parse_bool_option(preset.get("spatialglue_omics2_scale", True), default=True),
    )

    args = build_args(
        dataset_name=dataset_cfg.dataset_name,
        n_input=processed["omics1_full_np"].shape[1],
        n_input1=processed["omics2_full_np"].shape[1],
        n_clusters=n_clusters,
        random_seed=random_seed,
        epoch=dataset_cfg.epoch,
        lr=dataset_cfg.lr,
        sigma=dataset_cfg.sigma,
        loss_n=loss_n,
        loss_w=loss_w,
        loss_s=loss_s,
        loss_a=dataset_cfg.loss_a,
        early_stop_unsupervised=early_stop_unsupervised,
        early_stop_tool=early_stop_tool,
        early_stop_metric=early_stop_metric,
        early_stop_warmup_ratio=early_stop_warmup_ratio,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
        early_stop_sample_size=early_stop_sample_size,
        train_eval_interval=train_eval_interval,
        contrastive_start_ratio=dataset_cfg.contrastive_start_ratio,
        contrastive_ramp_ratio=dataset_cfg.contrastive_ramp_ratio,
        contrastive_min_scale=dataset_cfg.contrastive_min_scale,
        contrastive_max_scale=dataset_cfg.contrastive_max_scale,
        contrastive_alignment_power=dataset_cfg.contrastive_alignment_power,
        contrastive_balance_floor=dataset_cfg.contrastive_balance_floor,
        n_z=n_z,
        model_name=model_name,
        gcnii_alpha=gcnii_alpha,
        gcnii_lambda=gcnii_lambda,
        graph_hidden_dim=graph_hidden_dim,
        gcnii_num_layers=gcnii_num_layers,
        use_boundary_aware_gcn=use_boundary_aware_gcn,
        boundary_gate_temperature=boundary_gate_temperature,
        boundary_gate_min=boundary_gate_min,
    )
    args.fusion_mode = dataset_cfg.fusion_mode
    args.fusion_hidden_dim = dataset_cfg.fusion_hidden_dim
    args.use_dynamic_niche_graph = int(use_dynamic_niche_graph)
    args.dynamic_graph_warmup = dynamic_graph_warmup
    args.dynamic_graph_update_interval = dynamic_graph_update_interval
    args.dynamic_graph_smooth_beta = dynamic_graph_smooth_beta
    args.dynamic_graph_alpha_min = dynamic_graph_alpha_min
    args.dynamic_graph_adaptive_alpha = dynamic_graph_adaptive_alpha
    args.dynamic_graph_beta_min = dynamic_graph_beta_min
    args.dynamic_graph_adaptive_beta = dynamic_graph_adaptive_beta
    args.dynamic_graph_metric = dynamic_graph_metric
    args.dynamic_graph_source = dynamic_graph_source
    args.dynamic_graph_neighbors = int(niche_graph_neighbors)
    args.dynamic_graph_niche_weight = float(niche_graph_weight)

    fix_seed(random_seed)
    adj_train = preprocess_graph(processed["adj_mixed_dense_static"]).to(device)
    features_omics1 = torch.FloatTensor(processed["input_omics1_np"]).to(device)
    features_omics2 = torch.FloatTensor(processed["input_omics2_np"]).to(device)
    full_features_omics1 = torch.FloatTensor(processed["omics1_full_np"]).to(device)
    full_features_omics2 = torch.FloatTensor(processed["omics2_full_np"]).to(device)
    model = create_model(model_name, args).to(device)
    model = Train(
        args.epoch,
        model,
        features_omics1,
        features_omics2,
        adj_train,
        None,
        device,
        args,
        data_target=full_features_omics1,
        data_target1=full_features_omics2,
        adj_spatial=processed["spatial_adj_dense"],
        adj_niche=processed["niche_adj_dense_static"],
        dynamic_spatial_adj_dense=processed["spatial_adj_dense"],
        dynamic_niche_static_adj_dense=processed["niche_adj_dense_static"],
        dynamic_mixed_static_adj_dense=processed["adj_mixed_dense_static"],
    )

    dynamic_graph_state = getattr(model, "last_dynamic_graph", None) or {}
    final_mixed_adj_dense = dynamic_graph_state.get("mixed_adj_dense")
    adj_test = preprocess_graph(final_mixed_adj_dense).to(device) if final_mixed_adj_dense is not None else adj_train
    dynamic_graph_summary = {
        key: to_builtin(value)
        for key, value in dynamic_graph_state.items()
        if key not in {"mixed_adj_dense", "niche_adj_dense"}
    }

    primary_tool = dataset_cfg.primary_tool
    primary_scores = TestUnsupervised(
        model,
        processed["adata_omics1"],
        features_omics1,
        features_omics2,
        adj_test,
        device,
        args,
        primary_tool,
        data_target=full_features_omics1,
        data_target1=full_features_omics2,
        adj_spatial=processed["spatial_adj_dense"],
    )
    primary_last_test = copy.deepcopy(model.last_test)
    secondary_metrics = None
    if dataset_cfg.secondary_tool:
        secondary_scores = TestUnsupervised(
            model,
            processed["adata_omics1"],
            features_omics1,
            features_omics2,
            adj_test,
            device,
            args,
            dataset_cfg.secondary_tool,
            data_target=full_features_omics1,
            data_target1=full_features_omics2,
            adj_spatial=processed["spatial_adj_dense"],
        )
        secondary_metrics = {"tool": dataset_cfg.secondary_tool, **to_builtin(secondary_scores)}
        model.last_test = copy.deepcopy(primary_last_test)

    embedding = np.asarray(model.last_test["embedding"], dtype=np.float32)
    pred = np.asarray(model.last_test["pred"]).astype(str)
    adata_eval = processed["adata_omics1"].copy()
    adata_eval.obsm["X_spaNicheGraph"] = embedding
    adata_eval.obs["pred"] = pd.Categorical(pred.astype(str))
    try:
        sc.pp.neighbors(adata_eval, use_rep="X_spaNicheGraph", n_neighbors=30)
        sc.tl.umap(adata_eval, random_state=random_seed)
    except Exception as exc:  # noqa: BLE001 - UMAP is a convenience output, not a run gate.
        print(f"Skip UMAP for {run_tag}: {type(exc).__name__}: {exc}")

    primary_metrics = {"tool": primary_tool, **to_builtin(primary_scores)}
    metrics_payload = {
        "dataset": dataset_cfg.dataset_name,
        "experiment_preset": preset_name,
        "random_seed": int(random_seed),
        "manual_label_available": 0,
        "target_cluster_count_source": str(preset.get("target_cluster_count_source", "preset")),
        "n_clusters": int(n_clusters),
        "model_name": model_name,
        "full_pair_niche_basis": full_pair_niche_basis,
        "niche_graph_neighbors": int(niche_graph_neighbors),
        "niche_graph_max_cosine_distance": (
            None if niche_graph_max_cosine_distance is None else float(niche_graph_max_cosine_distance)
        ),
        "loss_n": float(loss_n),
        "loss_w": float(loss_w),
        "loss_s": float(loss_s),
        "loss_a": float(dataset_cfg.loss_a),
        "early_stop_unsupervised": int(early_stop_unsupervised),
        "early_stop_tool": early_stop_tool,
        "early_stop_metric": early_stop_metric,
        "early_stop_warmup_ratio": float(early_stop_warmup_ratio),
        "early_stop_patience": int(early_stop_patience),
        "early_stop_min_delta": float(early_stop_min_delta),
        "early_stop_sample_size": int(early_stop_sample_size),
        "train_eval_interval": int(train_eval_interval),
        "early_stop": to_builtin(getattr(model, "early_stop_state", {}) or {}),
        "n_z": int(n_z),
        "model_input": {
            "omics1_dim": int(args.n_input),
            "omics2_dim": int(args.n_input1),
        },
        "feature_representation": processed.get("feature_representation") or {},
        "graph_hidden_dim": int(graph_hidden_dim),
        "gcnii_num_layers": int(gcnii_num_layers),
        "gcnii_alpha": float(gcnii_alpha),
        "gcnii_lambda": float(gcnii_lambda),
        "use_boundary_aware_gcn": int(use_boundary_aware_gcn),
        "boundary_gate_temperature": float(boundary_gate_temperature),
        "boundary_gate_min": float(boundary_gate_min),
        "boundary_gate": to_builtin(model.last_test.get("boundary_gate") or {}),
        "primary_unsupervised": primary_metrics,
        "secondary_unsupervised": secondary_metrics,
        "branch_state": to_builtin(model.last_test.get("branch_state") or {}),
        "niche_feature": {
            "mode": processed.get("niche_feature_mode"),
            "feature_dim": None if processed.get("niche_features_np") is None else int(processed["niche_features_np"].shape[1]),
            "niche_graph_neighbors": processed.get("niche_graph_neighbors"),
            "niche_graph_max_cosine_distance": processed.get("niche_graph_max_cosine_distance"),
        },
        "dynamic_graph": dynamic_graph_summary,
    }

    spatial_coords = np.asarray(adata_eval.obsm["spatial"], dtype=np.float32)
    pred_df = pd.DataFrame(
        {
            "spot_id": adata_eval.obs_names.astype(str),
            "pred": pred.astype(str),
            "spatial1": spatial_coords[:, 0],
            "spatial2": spatial_coords[:, 1],
        }
    )

    if save_outputs:
        save_metrics_bundle(metrics_payload, dataset_result_dir, run_tag)
        pred_df.to_csv(dataset_result_dir / f"pred_labels_{run_tag}.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(embedding, index=adata_eval.obs_names.astype(str)).to_csv(
            dataset_result_dir / f"embedding_{run_tag}.csv",
            index_label="spot_id",
            encoding="utf-8-sig",
        )
        if save_heavy_outputs:
            save_adata_safe(adata_eval, dataset_result_dir / f"adata_plot_{run_tag}.h5ad")
        if save_heavy_outputs and dynamic_graph_state.get("mixed_adj_dense") is not None:
            np.save(dataset_result_dir / f"final_mixed_adj_{run_tag}.npy", np.asarray(dynamic_graph_state["mixed_adj_dense"], dtype=np.float32))
        if save_heavy_outputs and dynamic_graph_state.get("niche_adj_dense") is not None:
            np.save(dataset_result_dir / f"final_niche_adj_{run_tag}.npy", np.asarray(dynamic_graph_state["niche_adj_dense"], dtype=np.float32))

    if save_outputs:
        plot_unlabeled_outputs(
            adata_eval=adata_eval,
            run_tag=run_tag,
            figure_dir=figure_dir,
            dataset_cfg=dataset_cfg,
            save_outputs=True,
        )

    train_df = pd.DataFrame(getattr(model, "training_history", []))
    if not train_df.empty:
        if save_outputs:
            train_df.to_csv(dataset_result_dir / f"training_history_{run_tag}.csv", index=False, encoding="utf-8-sig")
        if save_outputs:
            fig, ax = plt.subplots(1, 1, figsize=(7, 4))
            ax.plot(train_df["epoch"], train_df["loss_total"], label="loss_total")
            for key in ["loss_ae", "loss_w", "loss_igae", "loss_adj", "loss_s", "loss_nce", "early_stop_score"]:
                if key in train_df:
                    ax.plot(train_df["epoch"], train_df[key], label=key)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Value")
            ax.legend(fontsize=8)
            ax.set_title(f"Training curves: {run_tag}")
            plt.tight_layout()
            save_figure(fig, figure_dir / f"training_curves_{run_tag}.png")

    return summarize_metrics_payload(metrics_payload)


def run_batch(dataset_cfg: DatasetDefaults, presets: dict[str, dict], parser_description: str):
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument("--presets", nargs="*", default=None, help="Preset names to run.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Random seeds to run.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Dataset directory.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT, help="Root directory for outputs.")
    parser.add_argument("--device", type=str, default="auto", help="Torch device, e.g. cpu, cuda:0, cuda:1.")
    parser.add_argument("--cuda", default=None, help="CUDA index shortcut, e.g. 0 or 1. Overrides --device.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs whose metrics json already exists.")
    parser.add_argument("--no-save", action="store_true", help="Run without saving outputs and figures.")
    parser.add_argument(
        "--skip-heavy-outputs",
        action="store_true",
        help="Do not save adata_plot h5ad or dense final adjacency npy files; metrics, predictions, embeddings, figures, and training history are still saved.",
    )
    parser.add_argument("--max-epochs-override", type=int, default=None, help="Temporarily override training epochs.")
    parser.add_argument(
        "--e15-rna-use-pca",
        choices=["auto", "true", "false", "1", "0", "yes", "no", "on", "off"],
        default="auto",
        help="Override E15 RNA representation: auto uses preset, true uses PCA, false uses scaled full RNA dimension.",
    )
    parser.add_argument("--e15-rna-pca-dim", type=int, default=None, help="Override requested E15 RNA PCA dimension.")
    args = parser.parse_args()

    if args.data_dir is not None:
        dataset_cfg = DatasetDefaults(**{**dataset_cfg.__dict__, "data_dir": args.data_dir})

    selected_presets = parse_presets(args.presets, presets)
    selected_seeds = parse_seeds(args.seeds)
    device = resolve_device(args.device, args.cuda)
    dataset_dir = ensure_dir(args.results_root / dataset_cfg.dataset_name)

    print(f"Running {dataset_cfg.dataset_name} presets: {selected_presets}")
    print(f"Seeds: {selected_seeds}")
    print(f"Results root: {args.results_root}")
    print(f"Device: {device}")

    summary_rows = []
    for preset_name in selected_presets:
        for seed in selected_seeds:
            run_tag = f"{preset_name}_{seed}"
            metrics_path = args.results_root / dataset_cfg.dataset_name / preset_name / f"metrics_{run_tag}.json"
            if args.skip_existing and metrics_path.exists():
                existing = load_existing_summary_row(metrics_path)
                if existing is not None:
                    summary_rows.append(existing)
                print(f"Skip existing: {metrics_path}")
                continue
            preset_to_run = dict(presets[preset_name])
            if args.e15_rna_use_pca != "auto":
                preset_to_run["e15_rna_use_pca"] = parse_bool_option(args.e15_rna_use_pca)
            if args.e15_rna_pca_dim is not None:
                preset_to_run["e15_rna_pca_dim"] = int(args.e15_rna_pca_dim)
            run_dataset_cfg = dataset_cfg
            if args.max_epochs_override is not None:
                run_dataset_cfg = DatasetDefaults(
                    **{**dataset_cfg.__dict__, "epoch": int(args.max_epochs_override)}
                )
            summary_rows.append(
                run_single_experiment(
                    dataset_cfg=run_dataset_cfg,
                    preset_name=preset_name,
                    preset=preset_to_run,
                    random_seed=seed,
                    results_root=args.results_root,
                    device=device,
                    save_outputs=not args.no_save,
                    save_figures=not args.no_save,
                )
            )

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        print(summary_df)


def run_unlabeled_multi_dataset_batch(
    dataset_cfgs: dict[str, DatasetDefaults],
    presets_by_dataset: dict[str, dict[str, dict]],
    parser_description: str,
):
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        choices=list(dataset_cfgs.keys()),
        help="Unlabeled datasets to run.",
    )
    parser.add_argument("--presets", nargs="*", default=None, help="Preset names to run. Names are dataset-local.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Random seeds to run.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override the dataset directory. This is supported when one dataset is configured.",
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT, help="Root directory for outputs.")
    parser.add_argument("--device", type=str, default="auto", help="Torch device, e.g. cpu, cuda:0, cuda:1.")
    parser.add_argument("--cuda", default=None, help="CUDA index shortcut, e.g. 0 or 1. Overrides --device.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs whose metrics json already exists.")
    parser.add_argument("--no-save", action="store_true", help="Run without saving outputs and figures.")
    parser.add_argument(
        "--skip-heavy-outputs",
        action="store_true",
        help="Do not save adata_plot h5ad or dense final adjacency npy files; metrics, predictions, embeddings, figures, and training history are still saved.",
    )
    parser.add_argument("--max-epochs-override", type=int, default=None, help="Temporarily override training epochs.")
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Override target cluster count for all selected datasets and presets.",
    )
    args = parser.parse_args()

    selected_datasets = list(args.datasets) if args.datasets else list(dataset_cfgs.keys())
    if args.data_dir is not None and len(selected_datasets) != 1:
        parser.error("--data-dir requires exactly one selected dataset.")
    selected_seeds = parse_seeds(args.seeds)
    device = resolve_device(args.device, args.cuda)

    print(f"Running unlabeled datasets: {selected_datasets}")
    print(f"Seeds: {selected_seeds}")
    print(f"Results root: {args.results_root}")
    print(f"Device: {device}")

    summary_rows = []
    for dataset_key in selected_datasets:
        dataset_cfg = dataset_cfgs[dataset_key]
        if args.data_dir is not None:
            dataset_cfg = DatasetDefaults(**{**dataset_cfg.__dict__, "data_dir": args.data_dir})
        presets = presets_by_dataset[dataset_key]
        selected_presets = parse_presets(args.presets, presets)
        print(f"Dataset {dataset_key} presets: {selected_presets}")
        for preset_name in selected_presets:
            for seed in selected_seeds:
                run_tag = f"{preset_name}_{seed}"
                metrics_path = args.results_root / dataset_cfg.dataset_name / preset_name / f"metrics_{run_tag}.json"
                if args.skip_existing and metrics_path.exists():
                    existing = load_existing_summary_row(metrics_path)
                    if existing is not None:
                        summary_rows.append(existing)
                    print(f"Skip existing: {metrics_path}")
                    continue
                preset_to_run = dict(presets[preset_name])
                if args.n_clusters is not None:
                    preset_to_run["n_clusters"] = int(args.n_clusters)
                    preset_to_run["target_cluster_count_source"] = "cli_override"
                run_dataset_cfg = dataset_cfg
                if args.max_epochs_override is not None:
                    run_dataset_cfg = DatasetDefaults(
                        **{**dataset_cfg.__dict__, "epoch": int(args.max_epochs_override)}
                    )
                summary_rows.append(
                    run_single_unlabeled_experiment(
                        dataset_cfg=run_dataset_cfg,
                        preset_name=preset_name,
                        preset=preset_to_run,
                        random_seed=seed,
                        results_root=args.results_root,
                        device=device,
                        save_outputs=not args.no_save,
                        save_heavy_outputs=not args.skip_heavy_outputs,
                    )
                )

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        print(summary_df)
