
"""
Train tiny CNN-LSTM on flows.npz using only CPU.
Requires: tensorflow-cpu 2.15.1, numpy, sklearn.
"""

import json, argparse, pathlib, os, time, numpy as np, tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder

# ───── argparse ──────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--data",   default=r"..\data\flows.npz")
ap.add_argument("--splits", default=r"..\splits.json")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch",  type=int, default=32)
ap.add_argument("--outdir", default="checkpoints")
args = ap.parse_args()

# ───── CPU tuning ───────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.config.threading.set_intra_op_parallelism_threads(0)   # “0” = all cores
tf.config.threading.set_inter_op_parallelism_threads(0)

# ───── Load tensors & splits ────────────────────────────────────────────
data = np.load(args.data)
X, y_raw = data["X"], data["y"]            # (N, win, features)
splits = json.load(open(args.splits))

le = LabelEncoder()
y_int = le.fit_transform(y_raw)            # e.g., [0,1,2,3]

def subset(idx): return X[idx], y_int[idx]

X_train, y_train = subset(splits["train"])
X_val,   y_val   = subset(splits["val"])
X_test,  y_test  = subset(splits["test"])

# ───── Class-weight for imbalance ───────────────────────────────────────
cw = compute_class_weight('balanced', classes=np.unique(y_int), y=y_int)
class_weight = dict(enumerate(cw))

# ───── Model definition ─────────────────────────────────────────────────
win, feats = X_train.shape[1:]
inputs = tf.keras.Input(shape=(win, feats, 1))            # add “channel” dim
x = tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu")(inputs)
x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
x = tf.keras.layers.Reshape((win, -1))(x)                # flatten spatial
x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
x = tf.keras.layers.LSTM(64)(x)
outputs = tf.keras.layers.Dense(len(le.classes_), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

# ───── Training ─────────────────────────────────────────────────────────
ckpt_dir = pathlib.Path(args.outdir)
ckpt_dir.mkdir(exist_ok=True)
ckpt_path = ckpt_dir / "cnn_lstm_best.h5"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(str(ckpt_path),
                                       save_best_only=True,
                                       monitor="val_accuracy",
                                       mode="max"),
    tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.3, verbose=1)
]

t0 = time.time()
history = model.fit(
    X_train[..., np.newaxis], y_train,
    validation_data=(X_val[..., np.newaxis], y_val),
    epochs=args.epochs,
    batch_size=args.batch,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=2
)
print("Training time:", (time.time() - t0)/60, "min")

# ───── Evaluation ───────────────────────────────────────────────────────
model.load_weights(ckpt_path)
loss, acc = model.evaluate(X_test[..., np.newaxis], y_test, verbose=0)
print("Test accuracy:", acc)
# save label encoder for later
np.save(ckpt_dir / "labels.npy", le.classes_)
