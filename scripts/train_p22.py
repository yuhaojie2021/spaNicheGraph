"""Train spaNicheGraph on the unlabeled P22 mouse brain RNA-ATAC dataset."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training import DatasetDefaults, run_unlabeled_multi_dataset_batch  # noqa: E402


COMMON = {
    "epoch": 260,
    "lr": 5e-5,
    "sigma": 1.0,
    "loss_w": 0.1,
    "loss_s": 0.1,
    "loss_a": 0.1,
    "fusion_mode": "concat",
    "fusion_hidden_dim": 32,
    "contrastive_start_ratio": 0.1,
    "contrastive_ramp_ratio": 0.2,
    "contrastive_min_scale": 0.0,
    "contrastive_max_scale": 1.0,
    "contrastive_alignment_power": 1.0,
    "contrastive_balance_floor": 0.25,
    "primary_tool": "kmeans",
    "secondary_tool": None,
    "spatial_neighbors": 3,
    "niche_neighbors": 6,
    "niche_graph_neighbors": 12,
}

DATASETS = {
    "P22": DatasetDefaults(
        dataset_name="P22",
        data_dir=PROJECT_ROOT / "datasets" / "P22",
        rna_filename="adata_RNA.h5ad",
        modality_filename="adata_peaks_normalized.h5ad",
        label_filename="",
        label_column="",
        modality_kind="ATAC",
        **COMMON,
    )
}

PRESETS = {
    "P22": {
        "spanichegraph_p22": {
            "model_name": "spanichepair_gcnii",
            "n_clusters": 11,
            "target_cluster_count_source": "RNA_clusters_nunique_in_input_obs",
            "loss_n": 0.0,
            "loss_w": 0.1,
            "loss_s": 0.1,
            "early_stop_unsupervised": 1,
            "early_stop_tool": "kmeans",
            "early_stop_metric": "silhouette",
            "early_stop_warmup_ratio": 0.5,
            "early_stop_patience": 5,
            "early_stop_min_delta": 1e-4,
            "early_stop_sample_size": 2000,
            "train_eval_interval": 20,
            "niche_graph_weight": 0.3,
            "use_niche_graph": True,
            "niche_feature_mode": "mean_std",
            "full_pair_niche_basis": "joint",
            "use_dynamic_niche_graph": True,
            "dynamic_graph_warmup": 0.3,
            "dynamic_graph_update_interval": 50,
            "dynamic_graph_smooth_beta": 0.05,
            "dynamic_graph_alpha_min": 0.05,
            "dynamic_graph_adaptive_alpha": 0,
            "dynamic_graph_beta_min": 0.01,
            "dynamic_graph_adaptive_beta": 1,
            "dynamic_graph_metric": "cosine",
            "dynamic_graph_source": "joint",
            "gcnii_alpha": 0.3,
            "gcnii_lambda": 0.4,
            "n_z": 40,
            "spatialglue_rna_pca_dim": 50,
            "spatialglue_prefer_existing_rna_pca": True,
            "spatialglue_omics2_pca_dim": 50,
            "spatialglue_prefer_existing_omics2_lsi": True,
            "spatialglue_omics2_use_clr": False,
            "spatialglue_omics2_scale": False,
        }
    }
}


if __name__ == "__main__":
    run_unlabeled_multi_dataset_batch(DATASETS, PRESETS, "Train spaNicheGraph on P22 mouse brain RNA-ATAC data.")
