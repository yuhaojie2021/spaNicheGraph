import numpy as np
import pandas as pd
import scanpy as sc


def pca(adata, use_reps=None, n_comps=20):
    from scipy.sparse.csc import csc_matrix
    from scipy.sparse.csr import csr_matrix
    from sklearn.decomposition import PCA

    reducer = PCA(n_components=int(n_comps))
    if use_reps is not None:
        return reducer.fit_transform(adata.obsm[use_reps])
    if isinstance(adata.X, (csc_matrix, csr_matrix)):
        return reducer.fit_transform(adata.X.toarray())
    return reducer.fit_transform(adata.X)


def mclust_R(adata, num_cluster, modelNames="EEE", used_obsm="X_spaNicheGraph_pca", random_seed=2020):
    """Run R mclust when rpy2 and mclust are available."""
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    import rpy2.robjects.numpy2ri

    robjects.r.library("mclust")
    rpy2.robjects.numpy2ri.activate()
    robjects.r["set.seed"](random_seed)
    rmclust = robjects.r["Mclust"]
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.asarray(res[-2])
    adata.obs["mclust"] = pd.Categorical(mclust_res.astype(int))
    return adata


# def search_res(adata, n_clusters, method="leiden", use_rep="X_spaNicheGraph", start=0.001, end=7.0, increment=0.001):
#     """Search a graph clustering resolution close to the requested cluster count."""
#     sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
#     best_res = float(start)
#     best_gap = float("inf")
#     best_count = None
#     for res in np.arange(float(start), float(end), float(increment)):
#         if method == "leiden":
#             sc.tl.leiden(adata, random_state=0, resolution=float(res))
#             count = int(pd.Series(adata.obs["leiden"]).nunique())
#         elif method == "louvain":
#             sc.tl.louvain(adata, random_state=0, resolution=float(res))
#             count = int(pd.Series(adata.obs["louvain"]).nunique())
#         else:
#             raise ValueError(f"Unsupported graph clustering method: {method}")
#         gap = abs(count - int(n_clusters))
#         if gap < best_gap:
#             best_gap = gap
#             best_res = float(res)
#             best_count = count
#         if count == int(n_clusters):
#             return float(res)
#     print(f"Resolution with exact {n_clusters} clusters not found; using {best_res:.4f} with {best_count} clusters.")
#     return best_res


def search_res(adata, n_clusters, method="leiden", use_rep="emb", start=0.01, end=3.0, increment=0.01, max_iters=100):
    """Search a graph clustering resolution close to the requested cluster count."""
    method = str(method).lower()
    target = int(n_clusters)
    left = float(start)
    right = float(end)
    tolerance = max(float(increment), 1e-6)
    max_iters = max(1, int(max_iters))
    evaluated = {}
    best_res = left
    best_count = None
    best_gap = float("inf")

    try:
        sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    except RuntimeError as exc:
        # Some older Scanpy/Numba combinations cannot initialize UMAP's cached
        # kernels. The Gaussian-connectivity path avoids that optional backend.
        if "cannot cache function" not in str(exc):
            raise
        sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep, method="gauss")

    def get_cluster_count(resolution):
        nonlocal best_res, best_count, best_gap
        resolution = round(float(np.clip(resolution, left, right)), 6)
        if resolution in evaluated:
            return evaluated[resolution]
        if method == "leiden":
            sc.tl.leiden(adata, random_state=0, resolution=resolution)
            count = int(pd.Series(adata.obs["leiden"]).nunique())
        elif method == "louvain":
            sc.tl.louvain(adata, random_state=0, resolution=resolution)
            count = int(pd.Series(adata.obs["louvain"]).nunique())
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
        evaluated[resolution] = count
        gap = abs(count - target)
        if gap < best_gap:
            best_res = resolution
            best_count = count
            best_gap = gap
        return count

    low, high = left, right
    for _ in range(max_iters):
        mid = (low + high) / 2.0
        count = get_cluster_count(mid)
        if count == target:
            return float(round(mid, 6))
        if count < target:
            low = mid
        else:
            high = mid
        if (high - low) < tolerance:
            break

    if best_count != target:
        print(f"Resolution with exact {target} clusters not found; using {best_res:.4f} with {best_count} clusters.")
    return float(best_res)


def clustering(
    adata,
    n_clusters=7,
    key="X_spaNicheGraph",
    add_key="spaNicheGraph",
    method="leiden",
    start=0.01,
    end=3.0,
    increment=0.01,
    use_pca=False,
    n_comps=20,
):
    """Cluster a latent representation stored in adata.obsm."""
    method = str(method).lower()
    used_key = key
    if use_pca:
        n_comps = max(2, min(int(n_comps), adata.obsm[key].shape[0] - 1, adata.obsm[key].shape[1]))
        used_key = key + "_pca"
        adata.obsm[used_key] = pca(adata, use_reps=key, n_comps=n_comps)

    if method == "mclust":
        adata = mclust_R(adata, used_obsm=used_key, num_cluster=n_clusters)
        adata.obs[add_key] = adata.obs["mclust"]
    elif method == "leiden":
        search_res(adata, n_clusters, use_rep=used_key, method=method, start=start, end=end, increment=increment)
        adata.obs[add_key] = adata.obs["leiden"]
    elif method == "louvain":
        search_res(adata, n_clusters, use_rep=used_key, method=method, start=start, end=end, increment=increment)
        adata.obs[add_key] = adata.obs["louvain"]
    else:
        raise ValueError(f"Unsupported clustering method: {method}")
    return adata
