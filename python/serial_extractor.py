"""
Read LiDAR scan data from the Arduino over serial and save each scan as a
text file under the chosen output directory. Each scan is a 2D distance grid:

    rows    = vertical positions (max_vert_angle - min_vert_angle) / increment
    columns = horizontal angles (1 .. 60 * hor_multiplier)

The file naming convention is `<prefix><N>.txt` where N auto-increments past
the highest existing index in the output directory.

Usage:
    python serial_extractor.py --port COM5 --label bottle --output ../data
"""
import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import serial


def find_next_index(folder_path: str, prefix: str) -> int:
    """Return one greater than the highest <prefix><N>.txt index in folder."""
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.txt$")
    highest = -1
    if os.path.isdir(folder_path):
        for fn in os.listdir(folder_path):
            m = pattern.match(fn)
            if m:
                highest = max(highest, int(m.group(1)))
    return highest + 1 if highest >= 0 else 1


def parse_pos(line: str):
    """Extract the integer following 'POS' on a serial line, or None."""
    if "POS" not in line:
        return None
    raw = line.split("POS")[-1].strip()
    try:
        return int(raw)
    except ValueError:
        print(f"Could not parse POS value: {raw!r}")
        return None


def parse_angle(line: str):
    """Parse 'Angle <a>: <value>' into (angle, value), or None on mismatch."""
    if "Angle" not in line:
        return None
    parts = line.split(":")
    if len(parts) != 2:
        return None
    try:
        angle = int(parts[0].split()[1])
        value = int(parts[1].strip())
    except (ValueError, IndexError):
        return None
    return angle, value


def collect_scan(ser: serial.Serial, *,
                 min_vert: int, max_vert: int, increment: int,
                 hor_multiplier: int) -> list:
    """Run a single tilt-sweep scan and return the resulting 2D distance grid."""
    width = 60 * hor_multiplier + 1
    height = (max_vert - min_vert + 1) // increment
    image = [[0] * width for _ in range(height)]

    ser.write(b"S")
    print("Sent start command 'S'.")

    pos_value = None
    while True:
        pos_value = parse_pos(ser.readline().decode("utf-8", errors="replace").strip())
        if pos_value is not None:
            print(f"POS: {pos_value}")
            if pos_value == min_vert + increment:
                break
    print("Started scan.")

    while pos_value < max_vert:
        line = ser.readline().decode("utf-8", errors="replace").strip()
        new_pos = parse_pos(line)
        if new_pos is not None:
            pos_value = new_pos
            print(f"POS: {pos_value}")
            continue

        parsed = parse_angle(line)
        if parsed is None:
            continue
        angle, value = parsed
        if 1 <= angle <= 60 * hor_multiplier:
            row = (pos_value - min_vert) // increment
            image[row][angle] = value

    return image


def save_scan(image: list, folder: str, prefix: str) -> str:
    """Persist the scan grid to disk and return the file path."""
    os.makedirs(folder, exist_ok=True)
    next_idx = find_next_index(folder, prefix)
    path = os.path.join(folder, f"{prefix}{next_idx}.txt")
    with open(path, "w") as f:
        for row in image:
            f.write(f"{row}\n")
    print(f"Saved scan: {path}")
    return path


def show_scan(image: list, clip_max: int = 1000) -> None:
    """Show the captured scan as a 2D depth image (interactive)."""
    arr = np.flipud(np.clip(np.array(image), None, clip_max))
    plt.imshow(arr, cmap="gray", interpolation="nearest")
    plt.colorbar()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM5", help="Serial port (e.g. COM5 or /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--label", default="data", help="Filename prefix / class label")
    parser.add_argument("--output", default="data", help="Output directory")
    parser.add_argument("--min-vert", type=int, default=720)
    parser.add_argument("--max-vert", type=int, default=1000)
    parser.add_argument("--increment", type=int, default=2)
    parser.add_argument("--hor-multiplier", type=int, default=1)
    parser.add_argument("--no-display", action="store_true", help="Skip the matplotlib preview")
    args = parser.parse_args()

    with serial.Serial(args.port, args.baud) as ser:
        image = collect_scan(
            ser,
            min_vert=args.min_vert,
            max_vert=args.max_vert,
            increment=args.increment,
            hor_multiplier=args.hor_multiplier,
        )

    save_scan(image, args.output, args.label)
    if not args.no_display:
        show_scan(image)


if __name__ == "__main__":
    main()
