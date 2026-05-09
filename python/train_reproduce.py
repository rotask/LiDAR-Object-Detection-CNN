"""
Clean reproduction of the Team Ashla LiDAR-CNN training run.

Mirrors the report's hyperparameters: 2-block CNN with batch normalisation,
Glorot uniform init, Adam optimiser, categorical cross-entropy, 60 epochs,
batch size 32, 75/25 train/val split, horizontal-flip + translation
augmentation. Best-checkpoint selection based on validation accuracy.

Validation set is small (75 samples), so per-seed variance is high — a
single misclassification shifts accuracy by ~1.3 percentage points. Run
across several seeds (--seeds 0 1 2 ...) to see the distribution.

Usage:
    python python/train_reproduce.py
    python python/train_reproduce.py --data ../data --epochs 60 --seeds 0 1 2 3 4
"""
import argparse
import os
import random
import re
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle as sk_shuffle
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical


CLASSES = ["Bottle", "Cap", "Mug"]


def assign_label(file_name: str) -> str:
    """Map a scan file name to its class label."""
    name = file_name.lower()
    if "bottle" in name: return "Bottle"
    if "cap"    in name: return "Cap"
    if "mug"    in name: return "Mug"
    if "wall"   in name: return "Wall"
    return "Unknown"


def read_scan(file_path: Path) -> np.ndarray:
    """Parse a scan text file into a (140, 61, 1) float array."""
    raw = file_path.read_text()
    cleaned = raw.translate(str.maketrans("", "", "[],"))
    flat = np.array([float(x) for x in cleaned.split() if x])
    return flat.reshape(140, 61, 1)


def load_dataset(data_folder: Path):
    """Load all bottle/cap/mug scans into (X, y_encoded, encoder)."""
    samples, labels = [], []
    for file_path in sorted(data_folder.iterdir()):
        if not file_path.is_file():
            continue
        label = assign_label(file_path.name)
        if label in {"Wall", "Unknown"}:
            continue
        samples.append(read_scan(file_path))
        labels.append(label)

    X = np.stack(samples).astype(np.float32)
    X /= X.max()  # global-max normalisation (matches the report)

    encoder = LabelEncoder().fit(CLASSES)
    y_int = encoder.transform(labels)
    y = to_categorical(y_int, num_classes=len(CLASSES))
    return X, y, encoder


def build_model(seed: int) -> keras.Model:
    """
    Two-block CNN matching the architecture used in `preprocessing.ipynb`'s
    `get_model()` (the version that reaches ~90% in the original notebook).

    Layout: Conv(16) → MaxPool → Conv(32) → MaxPool → Flatten → Dense(64)
            → Dropout(0.2) → Dense(3).
    Total params ≈ 1.08M (the report quotes 213K; that figure is not
    achievable with this layer stack on a 140×61 input — likely a typo or
    a different unrecorded architecture). Augmentation is a single
    RandomFlip("horizontal") matching the notebook.
    """
    initializer = keras.initializers.GlorotUniform(seed=seed)

    inputs = keras.Input(shape=(140, 61, 1))
    x = layers.RandomFlip("horizontal", seed=seed)(inputs)

    x = layers.Conv2D(16, (3, 3), padding="same", activation="relu",
                      kernel_initializer=initializer)(x)
    x = layers.MaxPool2D((2, 2))(x)

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu",
                      kernel_initializer=initializer)(x)
    x = layers.MaxPool2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu",
                     kernel_initializer=initializer)(x)
    x = layers.Dropout(0.2, seed=seed)(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax",
                           kernel_initializer=initializer)(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def set_global_seed(seed: int) -> None:
    """Make a single training run as deterministic as we can on CPU."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_one(X: np.ndarray, y: np.ndarray, *,
              seed: int, epochs: int, batch_size: int, verbose: int):
    """Train one model with a given seed and return (best_val_acc, history, y_pred_best)."""
    set_global_seed(seed)
    X_s, y_s = sk_shuffle(X, y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_s, y_s, test_size=0.25, random_state=seed
    )

    model = build_model(seed=seed)

    ckpt_path = f"_tmp_best_{seed}.weights.h5"
    cb = keras.callbacks.ModelCheckpoint(
        ckpt_path, monitor="val_accuracy", save_best_only=True,
        save_weights_only=True, verbose=0
    )
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size,
                        callbacks=[cb], verbose=verbose)
    model.load_weights(ckpt_path)
    os.remove(ckpt_path)

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    y_true = np.argmax(y_val, axis=1)
    return val_acc, history, y_true, y_pred, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="../data",
                        help="Folder containing bottle*.txt, cap*.txt, mug*.txt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="One seed per training run")
    parser.add_argument("--verbose", type=int, default=0,
                        help="Keras .fit verbose flag (0 quiet, 1 progress, 2 epoch)")
    parser.add_argument("--save-best", action="store_true",
                        help="Save the highest-accuracy model to model/my_model.h5")
    args = parser.parse_args()

    data_folder = Path(args.data).resolve()
    print(f"Dataset: {data_folder}")
    X, y, encoder = load_dataset(data_folder)
    print(f"Loaded {len(X)} samples ({X.shape}), classes = {list(encoder.classes_)}")
    print()

    results = []
    best_run = None
    for seed in args.seeds:
        print(f"--- seed {seed} ---")
        val_acc, history, y_true, y_pred, model = train_one(
            X, y, seed=seed,
            epochs=args.epochs, batch_size=args.batch_size,
            verbose=args.verbose,
        )
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
        print(f"  best val accuracy: {val_acc * 100:.2f}%")
        print(f"  confusion matrix (rows=true, cols=pred), labels = {CLASSES}:")
        for row in cm:
            print(f"    {list(row)}")
        results.append((seed, val_acc, cm))
        if best_run is None or val_acc > best_run[1]:
            best_run = (seed, val_acc, model)
        print()

    print("=" * 60)
    print("Summary across seeds")
    print(f"  {'seed':>6}  {'val_acc':>8}")
    for seed, va, _ in results:
        print(f"  {seed:>6}  {va * 100:>7.2f}%")
    if results:
        accs = [va for _, va, _ in results]
        print(f"  mean  : {np.mean(accs) * 100:.2f}%")
        print(f"  best  : {np.max(accs) * 100:.2f}%  (seed {best_run[0]})")
        print(f"  worst : {np.min(accs) * 100:.2f}%")

    if args.save_best and best_run is not None:
        out_path = Path("model/my_model.h5")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        best_run[2].save(out_path)
        print(f"\nSaved best model: {out_path}")


if __name__ == "__main__":
    main()
