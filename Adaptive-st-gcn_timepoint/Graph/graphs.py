import os
import numpy as np
import scipy.io as sio

# pip install mne
import mne

def get_coords_from_standard_1020(ch_names):
    montage = mne.channels.make_standard_montage("standard_1020")
    pos = montage.get_positions()["ch_pos"]

    coords, missing = [], []
    for ch in ch_names:
        if ch in pos:
            coords.append(pos[ch])
        else:
            missing.append(ch)

    if missing:
        raise ValueError(f"Channels missing in standard_1020 montage: {missing}")

    return np.stack(coords, axis=0)  # (C,3)

def pairwise_dist(coords):
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=-1)  # (C,C)

def dd_dense_from_dist(dist, sigma=0.25):
    A = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    np.fill_diagonal(A, 1.0)
    return A

def knn_from_dist(dist, k=4, sigma=0.25):
    C = dist.shape[0]
    W = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    np.fill_diagonal(W, 0.0)

    A = np.zeros((C, C), dtype=float)
    for i in range(C):
        nn = np.argsort(dist[i])[1:k+1]  # skip self
        A[i, nn] = W[i, nn]

    # make symmetric + diagonal
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 1.0)
    return A

def topk_from_weights(A, k=4):
    A = A.copy()
    np.fill_diagonal(A, 0.0)

    C = A.shape[0]
    keep = np.zeros_like(A, dtype=bool)
    for i in range(C):
        idx = np.argsort(A[i])[::-1][:k]
        keep[i, idx] = True

    keep = keep | keep.T
    A_sparse = A * keep
    np.fill_diagonal(A_sparse, 1.0)
    return A_sparse

def save_mat(path, adj):
    sio.savemat(path, {"adj": adj})
    print("Saved", path, adj.shape)

def main():
    out_dir = "graphs"
    os.makedirs(out_dir, exist_ok=True)

    # IMPORTANT: set this channel order to match your dataset exactly
    ch_names = [
        "Fp1","Fp2","F7","F3","Fz","F4","F8",
        "T7","C3","Cz","C4","T8",
        "P7","P3","Pz","P4","P8",
        "O1","O2"
    ]

    coords = get_coords_from_standard_1020(ch_names)
    dist = pairwise_dist(coords)

    # 1) identity (sanity)
    A_id = np.eye(len(ch_names), dtype=float)
    save_mat(os.path.join(out_dir, "adj_identity.mat"), A_id)

    # 2) random (debug)
    rng = np.random.default_rng(0)
    A_rand = rng.random((len(ch_names), len(ch_names)))
    A_rand = (A_rand + A_rand.T) / 2.0
    np.fill_diagonal(A_rand, 1.0)
    save_mat(os.path.join(out_dir, "adj_random.mat"), A_rand)

    # 3) DD dense
    A_dd = dd_dense_from_dist(dist, sigma=0.25)
    save_mat(os.path.join(out_dir, "adj_DD_dense.mat"), A_dd)

    # 4) KNN distance graphs
    for k in [4, 6, 8]:
        A_knn = knn_from_dist(dist, k=k, sigma=0.25)
        save_mat(os.path.join(out_dir, f"adj_DD_knn_k{k}.mat"), A_knn)

    # 5) top-k from DD dense weights
    for k in [4, 6, 8]:
        A_topk = topk_from_weights(A_dd, k=k)
        save_mat(os.path.join(out_dir, f"adj_DD_topk_k{k}.mat"), A_topk)

    print("\nDone. Pick any file in graphs/ as PathCfg['cheb'].")

if __name__ == "__main__":
    main()
