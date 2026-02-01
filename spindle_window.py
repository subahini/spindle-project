import numpy as np
from DE_PSD import DE_PSD

# -------------------------
# EDIT THESE PATHS
# -------------------------
X_PATH = "../data/processed_data/X_all.npy"# (N, C, T)
Y_PATH = "../data/processed_data/y_all.npy"       # (N,) or (N,1)
OUT_NPZ = "./data/spindle_DE_9bands.npz"

# -------------------------
# SETTINGS
# -------------------------
sfreq = 200
win_sec = 2
FOLDS = 5
SEED = 42

# 9 bands (same as paper)
fStart = [0.5, 2, 4, 6, 8, 11, 14, 22, 31]
fEnd   = [4,   6, 8, 11,14, 22, 31, 40, 50]

stft_para = {
    "stftn": 512,       # FFT points (can be 256/512)
    "fStart": fStart,
    "fEnd": fEnd,
    "fs": sfreq,
    "window": win_sec
}

def to_de_features(X_raw):
    # X_raw: (N, C, T)
    N, C, T = X_raw.shape
    X_feat = np.zeros((N, C, len(fStart)), dtype=np.float32)

    for i in range(N):
        # DE_PSD expects (C, T)
        _, de = DE_PSD(X_raw[i], stft_para)   # de: (C, 9)
        X_feat[i] = de.astype(np.float32)

    return X_feat

def make_folds(X, y, folds=5, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    splits = np.array_split(idx, folds)

    Fold_Data, Fold_Label, Fold_Num = [], [], []
    for s in splits:
        Fold_Data.append(X[s])
        yy = y[s].reshape(-1, 1).astype(np.float32)   # (n,1)
        Fold_Label.append(yy)
        Fold_Num.append(len(s))
    return np.array(Fold_Num), np.array(Fold_Data, dtype=object), np.array(Fold_Label, dtype=object)

if __name__ == "__main__":
    X_raw = np.load(X_PATH)   # (N,C,T)
    y = np.load(Y_PATH)
    y = y.reshape(-1).astype(int)

    print("Loaded:", X_raw.shape, y.shape, "pos_rate=", y.mean())

    print("Converting to 9-band DE features...")
    X_feat = to_de_features(X_raw)  # (N,C,9)

    Fold_Num, Fold_Data, Fold_Label = make_folds(X_feat, y, folds=FOLDS, seed=SEED)

    np.savez(OUT_NPZ,
             Fold_Num=Fold_Num,
             Fold_Data=Fold_Data,
             Fold_Label=Fold_Label)

    print("Saved:", OUT_NPZ)
    print("Example fold X:", Fold_Data[0].shape, "Y:", Fold_Label[0].shape)
