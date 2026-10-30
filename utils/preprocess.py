import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import scipy.sparse as sp
import torch
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.preprocessing import Normalizer
from sklearn.utils.extmath import randomized_svd
from typing import Optional


def to_numpy_array(matrix):
    if sp.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def pca(adata, use_reps=None, n_comps=10):
    """Run PCA on AnnData.X or an existing obsm representation."""
    from scipy.sparse.csc import csc_matrix
    from scipy.sparse.csr import csr_matrix
    from sklearn.decomposition import PCA

    reducer = PCA(n_components=int(n_comps))
    if use_reps is not None:
        return reducer.fit_transform(adata.obsm[use_reps])
    if isinstance(adata.X, (csc_matrix, csr_matrix)):
        return reducer.fit_transform(adata.X.toarray())
    return reducer.fit_transform(adata.X)


def clr_normalize_each_cell(adata, inplace=True):
    """Centered log-ratio normalization for ADT-like count vectors."""
    if not inplace:
        adata = adata.copy()

    def seurat_clr(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    adata.X = np.apply_along_axis(
        seurat_clr,
        1,
        adata.X.A if scipy.sparse.issparse(adata.X) else np.asarray(adata.X),
    )
    return adata


def _standardize_feature_matrix(features, eps=1e-6):
    features = np.asarray(features, dtype=np.float32)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return (features - mean) / np.maximum(std, eps)


def _neighbor_indices_from_coords(spatial_coords, n_neighbors=6, include_self=True):
    spatial_coords = np.asarray(spatial_coords, dtype=np.float32)
    extra = 1 if include_self else 0
    k = min(max(int(n_neighbors), 1) + extra, spatial_coords.shape[0])
    nbrs = NearestNeighbors(n_neighbors=k).fit(spatial_coords)
    indices = nbrs.kneighbors(return_distance=False)
    if not include_self and indices.shape[1] > 1:
        indices = indices[:, 1:]
    return [np.asarray(row, dtype=np.int64) for row in indices]


def _neighbor_indices_from_adjacency(spatial_adj, include_self=True):
    spatial_adj = to_numpy_array(spatial_adj).astype(np.float32, copy=False)
    if spatial_adj.ndim != 2 or spatial_adj.shape[0] != spatial_adj.shape[1]:
        raise ValueError("spatial_adj must be a square matrix.")
    neighbor_indices = []
    for row_idx in range(spatial_adj.shape[0]):
        idx = np.where(spatial_adj[row_idx] > 0.0)[0]
        if include_self and row_idx not in idx:
            idx = np.concatenate([idx, np.asarray([row_idx], dtype=np.int64)])
        if idx.size == 0:
            idx = np.asarray([row_idx], dtype=np.int64)
        neighbor_indices.append(idx.astype(np.int64, copy=False))
    return neighbor_indices


def _projection_matrix(n_features, projection_dim=16, random_seed=0):
    projection_dim = int(projection_dim or 0)
    if projection_dim <= 0 or n_features <= projection_dim:
        return None
    rng = np.random.default_rng(int(random_seed or 0))
    projection = rng.normal(
        loc=0.0,
        scale=1.0 / np.sqrt(float(projection_dim)),
        size=(n_features, projection_dim),
    )
    return projection.astype(np.float32)


def build_covariance_aware_niche_features(
    features,
    spatial_coords=None,
    spatial_adj=None,
    n_neighbors=6,
    include_self=True,
    add_std=True,
    covariance_projection_dim=16,
    covariance_include_diag=False,
    covariance_standardize=True,
    random_seed=0,
):
    """Build a compact local covariance-aware niche descriptor."""
    features = to_numpy_array(features).astype(np.float32, copy=False)
    if features.ndim != 2:
        raise ValueError("features must be a 2D matrix.")
    if spatial_adj is None and spatial_coords is None:
        raise ValueError("Either spatial_coords or spatial_adj must be provided.")

    if spatial_adj is not None:
        neighbor_indices = _neighbor_indices_from_adjacency(spatial_adj, include_self=include_self)
    else:
        spatial_coords = np.asarray(spatial_coords, dtype=np.float32)
        if features.shape[0] != spatial_coords.shape[0]:
            raise ValueError("features and spatial_coords must have the same number of spots.")
        neighbor_indices = _neighbor_indices_from_coords(spatial_coords, n_neighbors=n_neighbors, include_self=include_self)

    clean_features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    base_features = _standardize_feature_matrix(clean_features)
    projection = _projection_matrix(base_features.shape[1], projection_dim=covariance_projection_dim, random_seed=random_seed)
    projected = base_features if projection is None else base_features @ projection

    cov_dim = projected.shape[1]
    tri_k = 0 if covariance_include_diag else 1
    tri_rows, tri_cols = np.triu_indices(cov_dim, k=tri_k)
    descriptor_rows = []
    for idx in neighbor_indices:
        local_raw = clean_features[idx]
        local_projected = projected[idx]
        row_parts = [local_raw.mean(axis=0)]
        if add_std:
            row_parts.append(local_raw.std(axis=0))
        if local_projected.shape[0] <= 1:
            cov = np.zeros((cov_dim, cov_dim), dtype=np.float32)
        else:
            centered = local_projected - local_projected.mean(axis=0, keepdims=True)
            cov = centered.T @ centered / float(max(local_projected.shape[0] - 1, 1))
        row_parts.append(cov[tri_rows, tri_cols])
        descriptor_rows.append(np.concatenate(row_parts, axis=0))

    descriptor = np.stack(descriptor_rows, axis=0).astype(np.float32, copy=False)
    if covariance_standardize:
        descriptor = _standardize_feature_matrix(descriptor)
    return descriptor.astype(np.float32, copy=False)


def build_niche_features(
    features,
    spatial_coords,
    n_neighbors=6,
    include_self=True,
    add_std=True,
    covariance_mode="none",
    covariance_projection_dim=16,
    covariance_include_diag=False,
    covariance_standardize=True,
    random_seed=0,
):
    """Describe each spot by local multi-omics mean/std statistics."""
    covariance_mode = str(covariance_mode or "none").lower()
    if covariance_mode in {"covariance", "cov", "covariance_aware"}:
        return build_covariance_aware_niche_features(
            features,
            spatial_coords=spatial_coords,
            n_neighbors=n_neighbors,
            include_self=include_self,
            add_std=add_std,
            covariance_projection_dim=covariance_projection_dim,
            covariance_include_diag=covariance_include_diag,
            covariance_standardize=covariance_standardize,
            random_seed=random_seed,
        )

    features = to_numpy_array(features).astype(np.float32, copy=False)
    spatial_coords = np.asarray(spatial_coords, dtype=np.float32)
    if features.shape[0] != spatial_coords.shape[0]:
        raise ValueError("features and spatial_coords must have the same number of spots.")

    indices = _neighbor_indices_from_coords(spatial_coords, n_neighbors=n_neighbors, include_self=include_self)
    niche_mean = np.stack([features[idx].mean(axis=0) for idx in indices], axis=0)
    if add_std:
        niche_std = np.stack([features[idx].std(axis=0) for idx in indices], axis=0)
        return np.concatenate([niche_mean, niche_std], axis=1).astype(np.float32, copy=False)
    return niche_mean.astype(np.float32, copy=False)


def construct_niche_graph(
    niche_features,
    n_neighbors=12,
    metric="cosine",
    include_self=False,
    return_dense=False,
    max_distance=None,
):
    """Construct a KNN graph over local niche descriptors."""
    niche_features = np.asarray(niche_features, dtype=np.float32)
    if niche_features.ndim != 2:
        raise ValueError("niche_features must be a 2D array.")
    k = min(max(int(n_neighbors), 1), max(1, niche_features.shape[0] - (0 if include_self else 1)))
    if max_distance is None:
        graph = kneighbors_graph(niche_features, k, mode="connectivity", metric=metric, include_self=include_self)
    else:
        graph = kneighbors_graph(niche_features, k, mode="distance", metric=metric, include_self=include_self)
        graph.data = np.where(graph.data <= float(max_distance), 1.0, 0.0).astype(np.float32, copy=False)
        graph.eliminate_zeros()
    if return_dense:
        dense = graph.toarray().astype(np.float32)
        dense = dense + dense.T
        dense = np.where(dense > 0, 1.0, 0.0)
        np.fill_diagonal(dense, 0.0)
        return dense
    return graph


def construct_similarity_graph(features, n_neighbors=12, metric="cosine", include_self=False, return_dense=True):
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("features must be a 2D array.")
    max_neighbors = max(1, features.shape[0] - (0 if include_self else 1))
    k = min(max(int(n_neighbors), 1), max_neighbors)
    graph = kneighbors_graph(features, k, mode="connectivity", metric=metric, include_self=include_self)
    if not return_dense:
        return graph
    dense = graph.toarray().astype(np.float32)
    dense = dense + dense.T
    dense = np.where(dense > 0, 1.0, 0.0)
    np.fill_diagonal(dense, 0.0)
    return dense


def compute_graph_consistency(latent_adj, static_adj):
    latent_adj = to_numpy_array(latent_adj).astype(np.float32, copy=False)
    static_adj = to_numpy_array(static_adj).astype(np.float32, copy=False)
    if latent_adj.shape != static_adj.shape:
        raise ValueError("latent_adj and static_adj must have the same shape.")

    upper_mask = np.triu(np.ones(latent_adj.shape, dtype=bool), k=1)
    latent_values = latent_adj[upper_mask]
    static_values = static_adj[upper_mask]
    latent_edges = latent_values > 0.0
    static_edges = static_values > 0.0

    latent_count = int(latent_edges.sum())
    static_count = int(static_edges.sum())
    intersection = int(np.logical_and(latent_edges, static_edges).sum())
    union = int(np.logical_or(latent_edges, static_edges).sum())
    denom = float(np.linalg.norm(latent_values) * np.linalg.norm(static_values))
    return {
        "edge_overlap": float(intersection / latent_count) if latent_count > 0 else 0.0,
        "edge_jaccard": float(intersection / union) if union > 0 else 0.0,
        "weighted_cosine": float(np.dot(latent_values, static_values) / denom) if denom > 0.0 else 0.0,
        "latent_edge_count": latent_count,
        "static_edge_count": static_count,
        "edge_intersection": intersection,
        "edge_union": union,
    }


def update_dynamic_niche_graph(
    latent_features,
    static_niche_adj,
    n_neighbors=12,
    metric="cosine",
    smooth_beta=0.2,
    beta_min=0.02,
    adaptive_beta=True,
    return_stats=False,
):
    latent_adj = construct_similarity_graph(latent_features, n_neighbors=n_neighbors, metric=metric, include_self=False, return_dense=True)
    static_niche_adj = to_numpy_array(static_niche_adj).astype(np.float32, copy=False)
    beta_max = min(max(float(smooth_beta or 0.0), 0.0), 1.0)
    beta_min = min(max(float(beta_min or 0.0), 0.0), beta_max)
    consistency_stats = compute_graph_consistency(latent_adj, static_niche_adj)
    beta = beta_min + (beta_max - beta_min) * consistency_stats["edge_overlap"] if adaptive_beta else beta_max

    dynamic_adj = (1.0 - beta) * static_niche_adj + beta * latent_adj
    dynamic_adj = np.maximum(dynamic_adj, dynamic_adj.T)
    np.fill_diagonal(dynamic_adj, 0.0)
    dynamic_adj = dynamic_adj.astype(np.float32, copy=False)
    if return_stats:
        stats = {
            "version": "v2_fixed_alpha_adaptive_beta",
            "adaptive_beta": int(bool(adaptive_beta)),
            "beta": float(beta),
            "beta_min": float(beta_min),
            "beta_max": float(beta_max),
        }
        stats.update(consistency_stats)
        return dynamic_adj, stats
    return dynamic_adj


def blend_adjacency_matrices(spatial_adj, niche_adj, niche_weight=0.25, threshold=0.0):
    spatial_adj = to_numpy_array(spatial_adj).astype(np.float32, copy=False)
    niche_adj = to_numpy_array(niche_adj).astype(np.float32, copy=False)
    niche_weight = min(max(float(niche_weight or 0.0), 0.0), 1.0)
    mixed = (1.0 - niche_weight) * spatial_adj + niche_weight * niche_adj
    mixed = np.maximum(mixed, mixed.T)
    if threshold > 0.0:
        mixed = np.where(mixed >= threshold, mixed, 0.0)
    np.fill_diagonal(mixed, 0.0)
    return mixed.astype(np.float32, copy=False)


def construct_graph_by_coordinate(cell_position, n_neighbors=3):
    """Construct a spatial KNN graph from coordinates."""
    cell_position = np.asarray(cell_position, dtype=np.float32)
    n_spots = cell_position.shape[0]
    k = min(max(int(n_neighbors), 1) + 1, n_spots)
    nbrs = NearestNeighbors(n_neighbors=k).fit(cell_position)
    _, indices = nbrs.kneighbors(cell_position)
    used_neighbors = indices[:, 1:]
    x = np.repeat(np.arange(n_spots), used_neighbors.shape[1])
    y = used_neighbors.reshape(-1)
    return pd.DataFrame({"x": x, "y": y, "value": np.ones(x.size, dtype=np.float32)})


def transform_adjacent_matrix(adjacent):
    n_spot = int(max(adjacent["x"].max(), adjacent["y"].max()) + 1)
    return coo_matrix((adjacent["value"], (adjacent["x"], adjacent["y"])), shape=(n_spot, n_spot))


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def tfidf(X):
    """TF-IDF normalization for ATAC-like count matrices."""
    if scipy.sparse.issparse(X):
        row_sum = np.asarray(X.sum(axis=1)).reshape(-1)
        col_sum = np.asarray(X.sum(axis=0)).reshape(-1)
        tf = X.multiply(1.0 / np.maximum(row_sum, 1e-12)[:, None])
        idf = X.shape[0] / np.maximum(col_sum, 1e-12)
        return tf.multiply(idf)
    row_sum = np.asarray(X.sum(axis=1, keepdims=True))
    col_sum = np.asarray(X.sum(axis=0, keepdims=True))
    tf = X / np.maximum(row_sum, 1e-12)
    idf = X.shape[0] / np.maximum(col_sum, 1e-12)
    return tf * idf


def lsi(adata: anndata.AnnData, n_components: int = 20, use_highly_variable: Optional[bool] = None, **kwargs) -> None:
    """LSI analysis following the common TF-IDF plus SVD workflow."""
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    X_norm = Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    max_components = max(2, min(int(n_components), min(X_norm.shape) - 1))
    X_lsi = randomized_svd(X_norm, max_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= np.maximum(X_lsi.std(axis=1, ddof=1, keepdims=True), 1e-12)
    adata.obsm["X_lsi"] = X_lsi[:, 1:].astype(np.float32, copy=False)
