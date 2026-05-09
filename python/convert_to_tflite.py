"""
Convert a trained Keras model to TensorFlow Lite for deployment to the
Arduino Portenta H7.

The previous version of this file imported `train_model_updated.get_model()`
which no longer exists in the repo. This rewrite loads a saved Keras model
from disk, which decouples conversion from training.

Usage:
    python convert_to_tflite.py
    python convert_to_tflite.py --model my_model.h5 --output model.tflite

After conversion, build the C array for the Arduino sketch with:
    xxd -i model.tflite > arduino/sensor_data/model_data.cc
"""
import argparse
from pathlib import Path

import tensorflow as tf


def convert(model_path: Path, output_path: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model not found: {model_path}. Train and save one first "
            f"(e.g. via train_reproduce.py or preprocessing.ipynb)."
        )
    print(f"Loading Keras model: {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_bytes = converter.convert()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_bytes)
    size_kb = output_path.stat().st_size / 1024
    print(f"Saved: {output_path} ({size_kb:.1f} KB)")
    print()
    print("To embed into the Arduino sketch:")
    print(f"    xxd -i {output_path} > arduino/sensor_data/model_data.cc")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="model/my_model.h5",
                        help="Path to a saved Keras model (.h5 or .keras)")
    parser.add_argument("--output", default="model/model.tflite",
                        help="Where to write the TFLite model")
    args = parser.parse_args()
    convert(Path(args.model), Path(args.output))


if __name__ == "__main__":
    main()
