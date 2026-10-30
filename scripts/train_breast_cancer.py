"""Train spaNicheGraph on the single-omics Human Breast Cancer dataset."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from model.single_omics import SingleOmicsSpaNicheGraph  # noqa: E402
from utils.preprocess import (  # noqa: E402
    blend_adjacency_matrices,
    build_niche_features,
    construct_graph_by_coordinate,
    construct_niche_graph,
    preprocess_graph,
    pca,
    to_numpy_array,
    transform_adjacent_matrix,
    update_dynamic_niche_graph,
)

DEFAULT_DATASET_DIR = PROJECT_ROOT / "datasets" / "Human_Breast_Cancer"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "Human_Breast_Cancer"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_spatial_graph(coords: np.ndarray, n_neighbors: int) -> np.ndarray:
    graph = transform_adjacent_matrix(construct_graph_by_coordinate(coords, n_neighbors=n_neighbors)).toarray()
    graph = np.maximum(graph, graph.T).astype(np.float32)
    np.fill_diagonal(graph, 0.0)
    return graph


def load_and_preprocess(args: argparse.Namespace) -> dict:
    adata = sc.read_10x_h5(str(args.data_dir / "filtered_feature_bc_matrix.h5"))
    adata.var_names_make_unique()
    metadata = pd.read_csv(args.data_dir / "metadata.tsv", sep="\t", index_col=0)
    positions = pd.read_csv(args.data_dir / "spatial" / "tissue_positions_list.csv", header=None, index_col=0)
    positions = positions.loc[adata.obs_names]
    metadata = metadata.loc[adata.obs_names]

    sc.pp.filter_genes(adata, min_cells=int(args.min_cells))
    effective_hvg = min(int(args.hvg_top_genes), int(adata.n_vars))
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=effective_hvg)
    hvg_mask = adata.var["highly_variable"].to_numpy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    adata = adata[:, hvg_mask].copy()

    features = to_numpy_array(adata.X).astype(np.float32, copy=False)
    niche_pca_dim = min(int(args.niche_pca_dim), int(adata.n_obs - 1), int(adata.n_vars))
    if niche_pca_dim < 2:
        raise ValueError("niche_pca_dim must be at least 2 for niche graph construction.")
    niche_basis = pca(adata, n_comps=niche_pca_dim).astype(np.float32, copy=False)
    coords = positions.iloc[:, [4, 3]].to_numpy(dtype=np.float32)
    spatial_adj = make_spatial_graph(coords, args.spatial_neighbors)
    niche_features = build_niche_features(
        niche_basis,
        coords,
        n_neighbors=args.niche_neighbors,
        include_self=True,
        add_std=True,
        covariance_mode="mean_std",
        random_seed=args.seed,
    )
    niche_adj = construct_niche_graph(
        niche_features,
        n_neighbors=args.niche_graph_neighbors,
        metric="cosine",
        return_dense=True,
        max_distance=args.niche_graph_max_cosine_distance,
    )
    mixed_adj = blend_adjacency_matrices(spatial_adj, niche_adj, niche_weight=args.niche_graph_weight)
    labels = metadata[args.label_column].astype(str).to_numpy()
    print(f"Spots: {adata.n_obs}")
    print(f"RNA HVGs used directly as model input: {features.shape}")
    print(f"Niche PCA representation: {niche_basis.shape}")
    print(f"Label classes: {len(np.unique(labels))}")
    print(f"Spatial graph neighbors: {args.spatial_neighbors}")
    print(f"Niche feature shape: {niche_features.shape}")
    print(f"Niche graph neighbors: {args.niche_graph_neighbors}")
    print(f"Niche graph weight: {args.niche_graph_weight}")
    print(f"Niche graph max cosine distance: {args.niche_graph_max_cosine_distance}")
    return {
        "adata": adata,
        "features": features,
        "niche_basis": niche_basis,
        "coords": coords,
        "labels": labels,
        "spatial_adj": spatial_adj,
        "niche_features": niche_features,
        "niche_adj": niche_adj,
        "mixed_adj": mixed_adj,
        "metadata": metadata,
    }


def evaluate(embedding: np.ndarray, labels: np.ndarray, seed: int, n_clusters: int) -> tuple[dict, np.ndarray]:
    pred = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed).fit_predict(embedding)
    return {
        "nmi": float(normalized_mutual_info_score(labels, pred)),
        "ari": float(adjusted_rand_score(labels, pred)),
        "ami": float(adjusted_mutual_info_score(labels, pred)),
        "homogeneity": float(homogeneity_score(labels, pred)),
        "completeness": float(completeness_score(labels, pred)),
        "v_measure": float(v_measure_score(labels, pred)),
    }, pred


def edge_stats(adj: np.ndarray) -> dict:
    upper = np.triu(np.asarray(adj) > 0, k=1)
    n = int(adj.shape[0])
    possible = n * (n - 1) // 2
    edges = int(upper.sum())
    return {"n_nodes": n, "n_edges": edges, "edge_density": edges / possible if possible else 0.0}


def run(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    data = load_and_preprocess(args)
    device = torch.device(args.device)
    features = torch.as_tensor(data["features"], dtype=torch.float32, device=device)
    target = features
    static_adj = torch.as_tensor(preprocess_graph(data["mixed_adj"]).to(device).to_dense(), dtype=torch.float32)
    spatial_target = torch.as_tensor(preprocess_graph(data["spatial_adj"]).to(device).to_dense(), dtype=torch.float32)
    current_niche = data["niche_adj"].copy()
    current_mixed = data["mixed_adj"].copy()
    model = SingleOmicsSpaNicheGraph(
        features.shape[1],
        args.n_z,
        args.gcnii_alpha,
        args.gcnii_lambda,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    best_silhouette = -np.inf
    bad_count = 0
    updates = 0
    dynamic_stats = {}

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        x_hat, z_hat, adj_hat, z_ae, z_graph, embedding = model(features, static_adj)
        loss_ae = F.mse_loss(x_hat, target)
        loss_w = F.mse_loss(z_hat, target)
        loss_adj = F.binary_cross_entropy_with_logits(torch.mm(embedding, embedding.T), spatial_target)
        loss_s = F.mse_loss(z_graph, z_ae)
        loss = loss_ae + args.loss_w * loss_w + args.loss_a * loss_adj + args.loss_s * loss_s
        loss.backward()
        optimizer.step()
        row = {"epoch": epoch + 1, "loss_total": float(loss.item()), "loss_ae": float(loss_ae.item()), "loss_w": float(loss_w.item()), "loss_adj": float(loss_adj.item()), "loss_s": float(loss_s.item()), "dynamic_graph_updated": 0}

        with torch.no_grad():
            current_embedding = embedding.detach().cpu().numpy()

        if epoch % args.eval_interval == 0:
            log_metrics, _ = evaluate(
                current_embedding,
                data["labels"],
                args.seed,
                len(np.unique(data["labels"])),
            )
            print(
                f"Epoch_{epoch:3d} | loss {loss.item():.6f} | "
                f"nmi {log_metrics['nmi']:.4f} | ari {log_metrics['ari']:.4f} | "
                f"ami {log_metrics['ami']:.4f} | homogeneity {log_metrics['homogeneity']:.4f} | "
                f"completeness {log_metrics['completeness']:.4f} | "
                f"v_measure {log_metrics['v_measure']:.4f}"
            )
        if args.use_dynamic_graph and epoch + 1 >= args.dynamic_warmup_epoch and (epoch + 1 - args.dynamic_warmup_epoch) % args.dynamic_update_interval == 0:
            current_niche, current_mixed, dynamic_stats = _refresh_graph(current_embedding, data["spatial_adj"], current_niche, args)
            static_adj = torch.as_tensor(preprocess_graph(current_mixed).to(device).to_dense(), dtype=torch.float32)
            updates += 1
            row.update({"dynamic_graph_updated": 1, "dynamic_graph_edge_overlap": dynamic_stats.get("edge_overlap"), "dynamic_graph_edge_jaccard": dynamic_stats.get("edge_jaccard"), "dynamic_graph_beta": dynamic_stats.get("beta")})
            print(
                f"Dynamic graph update {updates} at epoch {epoch + 1} | "
                f"edge_overlap {dynamic_stats.get('edge_overlap', float('nan')):.4f} | "
                f"edge_jaccard {dynamic_stats.get('edge_jaccard', float('nan')):.4f} | "
                f"beta {dynamic_stats.get('beta', float('nan')):.4f}"
            )

        if args.early_stop and (epoch + 1) >= args.early_stop_warmup_epoch and (epoch + 1) % args.eval_interval == 0:
            eval_pred = KMeans(n_clusters=len(np.unique(data["labels"])), n_init=10, random_state=args.seed).fit_predict(current_embedding)
            score = float(silhouette_score(current_embedding, eval_pred, sample_size=min(args.sample_size, len(eval_pred)), random_state=args.seed))
            row["silhouette"] = score
            if score > best_silhouette + args.early_stop_min_delta:
                best_silhouette, bad_count = score, 0
            else:
                bad_count += 1
            if bad_count >= args.early_stop_patience:
                history.append(row)
                print(
                    f"Early stop at epoch {epoch + 1}: silhouette did not improve for "
                    f"{args.early_stop_patience} evaluations after warmup epoch "
                    f"{args.early_stop_warmup_epoch}."
                )
                break
        history.append(row)

    model.eval()
    with torch.no_grad():
        _, _, _, _, _, embedding = model(features, static_adj)
    embedding_np = embedding.cpu().numpy()
    metrics, pred = evaluate(embedding_np, data["labels"], args.seed, len(np.unique(data["labels"])))
    run_name = f"human_breast_cancer_single_{args.seed}"
    out_dir = args.output_dir / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"spot_id": data["adata"].obs_names, "ground_truth": data["labels"], "pred": pred, "x": data["coords"][:, 0], "y": data["coords"][:, 1]}).to_csv(out_dir / f"predictions_{run_name}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(embedding_np, index=data["adata"].obs_names).to_csv(out_dir / f"embedding_{run_name}.csv", index_label="spot_id")
    pd.DataFrame(history).to_csv(out_dir / f"training_history_{run_name}.csv", index=False)
    payload = {
        "dataset": "Human_Breast_Cancer",
        "task": "single_omics_spatial_domain_identification",
        "random_seed": args.seed,
        "n_spots": int(data["adata"].n_obs),
        "n_genes_after_filter": int(data["adata"].n_vars),
        "n_clusters": int(len(np.unique(data["labels"]))),
        "preprocessing": {
            "min_cells": args.min_cells,
            "hvg_top_genes": args.hvg_top_genes,
            "normalization": "normalize_total_1e4_log1p_scale",
            "model_input_use_pca": 0,
            "model_input_dim": int(data["features"].shape[1]),
            "niche_basis_use_pca": 1,
            "niche_pca_dim": int(data["niche_basis"].shape[1]),
        },
        "model": {
            "name": "single_omics_spanichegraph",
            "n_z": args.n_z,
            "gcnii_alpha": args.gcnii_alpha,
            "gcnii_lambda": args.gcnii_lambda,
        },
        "graph": {"spatial": edge_stats(data["spatial_adj"]), "niche_static": edge_stats(data["niche_adj"]), "mixed_static": edge_stats(data["mixed_adj"]), "niche_final": edge_stats(current_niche), "mixed_final": edge_stats(current_mixed)},
        "dynamic_graph": {"enabled": bool(args.use_dynamic_graph), "warmup_epoch": args.dynamic_warmup_epoch, "update_interval": args.dynamic_update_interval, "updates_applied": updates, **{k: v for k, v in dynamic_stats.items() if k in {"edge_overlap", "edge_jaccard", "beta", "beta_min", "beta_max"}}},
        "training": {"epochs_requested": args.epochs, "epochs_completed": len(history), "early_stop": bool(args.early_stop), "early_stop_patience": args.early_stop_patience},
        "primary": {"tool": "kmeans", **metrics},
    }
    (out_dir / f"metrics_{run_name}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["primary"], indent=2))
    print(f"Saved single-omics result to {out_dir}")
    return payload


def _refresh_graph(embedding, spatial_adj, static_niche, args):
    dynamic_niche, stats = update_dynamic_niche_graph(embedding, static_niche, n_neighbors=args.niche_graph_neighbors, metric="cosine", smooth_beta=args.dynamic_smooth_beta, beta_min=args.dynamic_beta_min, adaptive_beta=True, return_stats=True)
    mixed = blend_adjacency_matrices(spatial_adj, dynamic_niche, niche_weight=args.niche_graph_weight)
    return dynamic_niche, mixed, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train single-omics spaNicheGraph on Human Breast Cancer Visium data.")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--label-column", default="ground_truth", choices=("ground_truth", "fine_annot_type", "annot_type"))
    parser.add_argument("--min-cells", type=int, default=10)
    parser.add_argument("--hvg-top-genes", type=int, default=3000)
    parser.add_argument("--niche-pca-dim", type=int, default=50)
    parser.add_argument("--spatial-neighbors", type=int, default=3)
    parser.add_argument("--niche-neighbors", type=int, default=6)
    parser.add_argument("--niche-graph-neighbors", type=int, default=12)
    parser.add_argument("--niche-graph-weight", type=float, default=0.3)
    parser.add_argument("--niche-graph-max-cosine-distance", type=float, default=None)
    parser.add_argument("--n-z", type=int, default=40)
    parser.add_argument("--gcnii-alpha", type=float, default=0.3)
    parser.add_argument("--gcnii-lambda", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss-w", type=float, default=0.1)
    parser.add_argument("--loss-a", type=float, default=0.1)
    parser.add_argument("--loss-s", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=120)
    dynamic_group = parser.add_mutually_exclusive_group()
    dynamic_group.add_argument(
        "--use-dynamic-graph",
        dest="use_dynamic_graph",
        action="store_true",
        help="Enable training-time latent niche-graph refinement (default).",
    )
    dynamic_group.add_argument(
        "--no-dynamic-graph",
        dest="use_dynamic_graph",
        action="store_false",
        help="Disable training-time latent niche-graph refinement.",
    )
    parser.set_defaults(use_dynamic_graph=True)
    parser.add_argument("--dynamic-warmup-epoch", type=int, default=100)
    parser.add_argument("--dynamic-update-interval", type=int, default=50)
    parser.add_argument("--dynamic-smooth-beta", type=float, default=0.05)
    parser.add_argument("--dynamic-beta-min", type=float, default=0.01)
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--early-stop-warmup-epoch", type=int, default=150)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--sample-size", type=int, default=2000)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
