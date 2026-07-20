"""Convert official C2Seg-AB full-scene MAT files to TIFF."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import tifffile


BRIGHT_COLORS = [
    (0, 0, 0),
    (180, 180, 180),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 190),
    (0, 128, 128),
    (230, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (128, 128, 0),
    (255, 215, 180),
    (0, 0, 128),
]


def hsv_to_rgb_uint8(hue: int, saturation: float, value: float) -> tuple[int, int, int]:
    c = value * saturation
    x = c * (1 - abs((hue / 60) % 2 - 1))
    m = value - c
    if hue < 60:
        r, g, b = c, x, 0
    elif hue < 120:
        r, g, b = x, c, 0
    elif hue < 180:
        r, g, b = 0, c, x
    elif hue < 240:
        r, g, b = 0, x, c
    elif hue < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)


def build_palette(num_classes: int = 256) -> np.ndarray:
    colors = []
    for class_id in range(num_classes):
        if class_id < len(BRIGHT_COLORS):
            colors.append(BRIGHT_COLORS[class_id])
        else:
            colors.append(hsv_to_rgb_uint8((class_id * 47) % 360, 0.82, 1.0))
    return np.asarray(colors, dtype=np.uint8)


def save_tif(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bytes_total = math.prod(array.shape) * array.dtype.itemsize
    tifffile.imwrite(path, array, bigtiff=bytes_total > 3_800_000_000)
    print(f"wrote {path.name}: shape={array.shape}, dtype={array.dtype}", flush=True)


def load_variable(path: Path, key: str) -> np.ndarray:
    data = loadmat(path, variable_names=[key])
    if key not in data:
        raise KeyError(f"{path} does not contain {key}")
    return np.asarray(data[key])


def convert_scene(mat_path: Path, output_dir: Path) -> None:
    scene = mat_path.stem.replace("_multimodal", "").lower()
    for key in ("MSI", "SAR", "HSI"):
        arr = load_variable(mat_path, key)
        if arr.ndim != 3:
            raise ValueError(f"{mat_path}:{key} expected 3D, got {arr.shape}")
        arr = np.transpose(arr, (2, 0, 1))
        save_tif(output_dir / f"{scene}_{key}.tif", arr)
        del arr

    label = load_variable(mat_path, "label")
    if label.ndim != 2:
        label = np.squeeze(label)
    save_tif(output_dir / f"{scene}_label.tif", label)
    palette = build_palette(256)
    label_index = np.where(label < 0, 0, label).astype(np.int64) % 256
    color = palette[label_index]
    save_tif(output_dir / f"{scene}_label_color.tif", color)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for mat_path in sorted(args.mat_dir.glob("*_multimodal.mat")):
        convert_scene(mat_path, args.output_dir)


if __name__ == "__main__":
    main()
