"""
Visualise a LiDAR scan saved by serial_extractor.py.

The scan file is a list-of-lists text dump where each row is a Python-style
list of integer distances (one row per vertical tilt, one column per
horizontal angle).

Usage:
    python visualize.py path/to/scan.txt           # 2D depth image
    python visualize.py path/to/scan.txt --mode 3d # 3D surface plot
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_depth_image(file_path: str) -> np.ndarray:
    """Parse a scan file into a 2D numpy array of distances."""
    with open(file_path, "r") as f:
        lines = [line.strip().strip("[]") for line in f if line.strip()]
    rows = [list(map(int, line.split(","))) for line in lines if line]
    return np.array(rows)


def normalise(image: np.ndarray, clip_max: int = 2000) -> np.ndarray:
    """Clip out-of-range distances and rescale to [0, 1] (vertically flipped for plotting)."""
    arr = np.clip(image, None, clip_max).astype(float)
    rng = arr.max() - arr.min()
    if rng == 0:
        return np.zeros_like(arr)
    return np.flipud((arr - arr.min()) / rng)


def plot_2d(image: np.ndarray) -> None:
    plt.imshow(image, cmap="gray", interpolation="nearest")
    plt.colorbar()
    plt.title("LiDAR depth image")
    plt.show()


def plot_3d(image: np.ndarray) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(x, y, image, cmap="viridis", edgecolor="none")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title("LiDAR depth surface")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", help="Path to scan file (e.g. data/bottle1.txt)")
    parser.add_argument("--mode", choices=["2d", "3d"], default="2d")
    parser.add_argument("--clip-max", type=int, default=2000,
                        help="Clip distances above this value before normalising")
    args = parser.parse_args()

    image = normalise(load_depth_image(args.file), clip_max=args.clip_max)
    (plot_2d if args.mode == "2d" else plot_3d)(image)


if __name__ == "__main__":
    main()
