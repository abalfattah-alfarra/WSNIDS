"""
Prune and post-train quantise cnn_lstm_best.h5 → cnn_lstm_int8.tflite
Requires: tensorflow-model-optimization
pip install tensorflow-model-optimization==0.8.0
"""
import tensorflow as tf, pathlib, argparse
import tensorflow_model_optimization as tfmot
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt",   default="checkpoints/cnn_lstm_best.h5")
ap.add_argument("--labels", default="checkpoints/labels.npy")
ap.add_argument("--out",    default="checkpoints/cnn_lstm_int8.tflite")
args = ap.parse_args()

model = tf.keras.models.load_model(args.ckpt)

# 50 % magnitude pruning (fine-tune 1 epoch on val set for speed)
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
prune_model = prune_low_magnitude(model, pruning_schedule=tfmot.sparsity.keras.ConstantSparsity(0.5, 0))
prune_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
prune_model.fit(np.zeros((1, *model.input.shape[1:])), np.zeros((1,)), epochs=1, verbose=0)

# Strip pruning wrappers
prune_model = tfmot.sparsity.keras.strip_pruning(prune_model)

# INT8 quantisation
converter = tf.lite.TFLiteConverter.from_keras_model(prune_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
pathlib.Path(args.out).write_bytes(tflite_model)
print("Saved", args.out, "size", len(tflite_model)//1024, "KB")
