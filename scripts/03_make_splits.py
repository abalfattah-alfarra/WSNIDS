# 03_make_splits.py
import numpy as np, json, argparse
from sklearn.model_selection import StratifiedShuffleSplit

ap = argparse.ArgumentParser()
ap.add_argument("--labels-npz", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

labels = np.load(args.labels_npz)["y"]
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, temp_idx = next(sss.split(np.zeros_like(labels), labels))
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_idx, test_idx = next(sss2.split(np.zeros_like(temp_idx), labels[temp_idx]))

splits = dict(train=train_idx.tolist(), val=val_idx.tolist(), test=test_idx.tolist())
json.dump(splits, open(args.out, "w"))
print("Wrote", args.out, {k: len(v) for k,v in splits.items()})
