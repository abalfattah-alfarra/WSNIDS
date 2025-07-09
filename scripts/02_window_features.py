import argparse, pandas as pd, numpy as np, re

def normalize(col):
    """strip, lower, replace spaces/tabs with underscore"""
    return re.sub(r"\s+", "_", col.strip().lower())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True,  help="Parquet file (packets)")
    ap.add_argument("--out",    required=True,  help=".npz output (X,y)")
    ap.add_argument("--label",  default="attack_type", help="Label column name after normalizing")
    ap.add_argument("--win",    type=int, default=5, help="Sliding window size")
    args = ap.parse_args()

    # ── 1) read and sanitize headers ────────────────────────────────────────────
    df = pd.read_parquet(args.input)
    df.columns = [normalize(c) for c in df.columns]

    label = normalize(args.label)
    if label not in df.columns:
        raise SystemExit(f"Label column '{label}' not found. "
                         f"Columns after normalization:\n{df.columns.tolist()}")

    # ── 2) choose numeric feature columns (exclude label) ───────────────────────
    feat_cols = [c for c in df.columns if c != label and pd.api.types.is_numeric_dtype(df[c])]
    print("Feature columns:", feat_cols)

    # ── 3) build sliding-window tensors ─────────────────────────────────────────
    X, y, w = [], [], args.win
    for i in range(len(df) - w + 1):
        sub = df.iloc[i : i + w]
        if len(sub[label].unique()) == 1:                  # same label in window
            X.append(sub[feat_cols].values)
            y.append(sub[label].iloc[0])

    X = np.stack(X).astype(np.float32)                     # (N, win, |feat|)
    np.savez_compressed(args.out, X=X, y=np.array(y))

    print(f"Saved {args.out}  —  shape {X.shape}   "
          f"class counts {np.unique(y, return_counts=True)}")

if __name__ == "__main__":
    main()
