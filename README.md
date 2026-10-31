# spaNicheGraph

spaNicheGraph is an unsupervised graph-learning framework for spatial domain
identification. It combines a physical spatial graph with a multi-omics niche
graph, preserves spot-level molecular attributes through an attribute-anchored
graph encoder (AAGE), and refines the niche graph during training.

This repository contains the complete model and the four dataset-level training
entry points used in the manuscript. Ablation variants, external baselines,
downstream biological analyses, manuscript figures, datasets, and trained
outputs are intentionally excluded.


## Repository structure

```text
model/                      Core paired- and single-omics models
train/                      Losses, optimization, clustering, and evaluation
utils/                      Graph construction and preprocessing utilities
scripts/train_a1.py         A1 human lymph node RNA-ADT experiment
scripts/train_e15.py        E15 mouse embryo RNA-ATAC experiment
scripts/train_p22.py        P22 mouse brain RNA-ATAC experiment
scripts/train_breast_cancer.py  Human breast cancer spatial transcriptomics
training.py                 Shared multimodal data and training workflow
```

## Installation

Python 3.8-3.11 is supported. Install a PyTorch build compatible with
your CUDA driver, then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For CPU-only use, install the appropriate CPU build of PyTorch before running
the same requirements command.

## Data layout

Datasets are not redistributed. Place the processed inputs under `datasets/`
as shown below, or pass `--data-dir` to an individual training script.

- A1 and P22: SpatialGlue Zenodo record
  [10362607](https://zenodo.org/records/10362607). The A1 data are also
  available from GEO under accession `GSE263617`.
- E15: MISAR-seq Zenodo record
  [7714382](https://doi.org/10.5281/zenodo.7714382).
- Human breast cancer: [10x Genomics Human Breast Cancer, Block A,
  Section 1](https://www.10xgenomics.com/datasets/human-breast-cancer-block-a-section-1-1-standard-1-1-0).

```text
datasets/
├── A1/
│   ├── adata_RNA.h5ad
│   ├── adata_ADT.h5ad
│   └── labels.csv
├── E15/
│   ├── rna.h5ad
│   ├── atac.h5ad
│   └── labels.csv
├── P22/
│   ├── adata_RNA.h5ad
│   └── adata_peaks_normalized.h5ad
└── Human_Breast_Cancer/
    ├── filtered_feature_bc_matrix.h5
    ├── metadata.tsv
    └── spatial/
        └── tissue_positions_list.csv
```

Both AnnData objects in a paired dataset must contain the same spots in the
same order and an `obsm["spatial"]` coordinate matrix. The P22 inputs used in
the manuscript additionally contain `obsm["X_pca"]` in the RNA object and
`obsm["X_lsi"]` in the ATAC object. Labels are used for evaluation and target
cluster count, not as supervision in the representation-learning loss.

## Training

The manuscript experiments used seeds 2022-2026. The multimodal entry points
accept one or more seeds in a single command:

```bash
python scripts/train_a1.py --seeds 2022 2023 2024 2025 2026 --device cuda:0
python scripts/train_e15.py --seeds 2022 2023 2024 2025 2026 --device cuda:0
python scripts/train_p22.py --seeds 2022 2023 2024 2025 2026 --device cuda:0
```

The single-omics breast cancer entry point runs one seed at a time:

```bash
for seed in 2022 2023 2024 2025 2026; do
  python scripts/train_breast_cancer.py --seed "$seed" --device cuda:0
done
```

Use `--help` on any entry point for path and runtime overrides. Multimodal
outputs are written below `results/<dataset>/<preset>/`; breast cancer outputs
are written below `results/Human_Breast_Cancer/seed_<seed>/`.

## Main experiment settings

| Dataset | Modalities | Epochs | Latent dim. | AAGE alpha | Lambda | Clustering |
|---|---|---:|---:|---:|---:|---|
| A1 | RNA + ADT | 300 | 40 | 0.3 | 0.4 | K-means |
| E15 | RNA + ATAC | 300 | 40 | 0.5 | 0.3 | Leiden |
| P22 | RNA + ATAC | 260 | 40 | 0.3 | 0.4 | K-means |
| Human breast cancer | RNA | 120 | 40 | 0.3 | 0.4 | K-means |

All four entry points use a spatial-neighbor count of 3, a local niche-neighbor
count of 6, a niche-graph neighbor count of 12, and a niche-graph mixing weight
of 0.3. Dataset-specific preprocessing is encoded in the corresponding script
and `training.py`.

## Reproducibility notes

- Dynamic graph refinement is enabled in all four complete-model entry points.
- The supervised labels are not passed to the neural-network loss or graph
  refinement procedure.
- GPU, CUDA, PyTorch, sparse linear algebra, Leiden, and K-means versions can
  cause small numerical differences. Record the environment and random seed
  with every run.
- Large dense adjacency matrices may require substantial host and GPU memory.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
