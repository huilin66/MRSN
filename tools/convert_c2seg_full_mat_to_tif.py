"""Convert C2Seg full-scene MATLAB v7.3 files to TIFF.

The official C2Seg-BW full-scene files are MATLAB v7.3/HDF5 files:
    beijing.mat, wuhan.mat, beijing_label.mat, wuhan_label.mat

This script writes band-first TIFF stacks for image modalities and writes both
gray-index and RGB pseudo-color TIFFs for labels.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import h5py
import numpy as np
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


def estimate_bytes(shape: tuple[int, ...], dtype: np.dtype) -> int:
    return math.prod(shape) * np.dtype(dtype).itemsize


def rows_per_block(shape: tuple[int, ...], dtype: np.dtype, target_mb: int) -> int:
    if len(shape) == 2:
        row_bytes = shape[1] * np.dtype(dtype).itemsize
        height = shape[0]
    elif len(shape) == 3:
        row_bytes = shape[0] * shape[2] * np.dtype(dtype).itemsize
        height = shape[1]
    else:
        raise ValueError(f"Unsupported dataset shape: {shape}")
    return max(1, min(height, (target_mb * 1024 * 1024) // max(row_bytes, 1)))


def create_memmap(path: Path, shape: tuple[int, ...], dtype: np.dtype):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    bigtiff = estimate_bytes(shape, dtype) > 3_800_000_000
    return tifffile.memmap(path, shape=shape, dtype=dtype, bigtiff=bigtiff)


def convert_dataset(ds, output_path: Path, block_mb: int) -> None:
    shape = tuple(int(v) for v in ds.shape)
    dtype = np.dtype(ds.dtype)
    print(f"write {output_path.name}: shape={shape}, dtype={dtype}", flush=True)
    out = create_memmap(output_path, shape, dtype)

    if len(shape) == 2:
        step = rows_per_block(shape, dtype, block_mb)
        for row in range(0, shape[0], step):
            end = min(row + step, shape[0])
            out[row:end, :] = ds[row:end, :]
            print(f"  rows {row}:{end}", flush=True)
    elif len(shape) == 3:
        step = rows_per_block(shape, dtype, block_mb)
        for row in range(0, shape[1], step):
            end = min(row + step, shape[1])
            out[:, row:end, :] = ds[:, row:end, :]
            print(f"  rows {row}:{end}", flush=True)
    else:
        raise ValueError(f"Unsupported dataset shape: {shape}")

    out.flush()
    del out


def write_label_color(ds, output_path: Path, block_mb: int) -> None:
    shape = tuple(int(v) for v in ds.shape)
    if len(shape) != 2:
        raise ValueError(f"Expected 2D label, got {shape}")

    print(f"write {output_path.name}: shape=({shape[0]}, {shape[1]}, 3), dtype=uint8", flush=True)
    out_shape = (shape[0], shape[1], 3)
    out = create_memmap(output_path, out_shape, np.dtype("uint8"))
    palette = build_palette(256)
    step = max(1, min(shape[0], (block_mb * 1024 * 1024) // max(shape[1] * 3, 1)))

    for row in range(0, shape[0], step):
        end = min(row + step, shape[0])
        labels = np.asarray(ds[row:end, :])
        color_index = np.where(labels < 0, 0, labels).astype(np.int64)
        color_index = np.mod(color_index, 256)
        out[row:end, :, :] = palette[color_index]
        print(f"  color rows {row}:{end}", flush=True)

    out.flush()
    del out


def convert_label_file(path: Path, output_dir: Path, block_mb: int) -> None:
    stem = path.stem
    with h5py.File(path, "r") as src:
        ds = src["label"]
        convert_dataset(ds, output_dir / f"{stem}.tif", block_mb)
        write_label_color(ds, output_dir / f"{stem}_color.tif", block_mb)


def convert_scene_file(path: Path, output_dir: Path, block_mb: int) -> None:
    scene = path.stem
    with h5py.File(path, "r") as src:
        for key in ("MSI", "SAR", "HSI"):
            if key in src:
                convert_dataset(src[key], output_dir / f"{scene}_{key}.tif", block_mb)
        if "label" in src:
            ds = src["label"]
            convert_dataset(ds, output_dir / f"{scene}_label_rawcode.tif", block_mb)
            write_label_color(ds, output_dir / f"{scene}_label_rawcode_color.tif", block_mb)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src_dir",
        default=os.environ.get("C2SEG_FULL_BW_ROOT", "."),
        help="Directory containing beijing.mat, wuhan.mat, beijing_label.mat, and wuhan_label.mat.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for TIFF outputs. Defaults to <src_dir>/tif.",
    )
    parser.add_argument(
        "--block_mb",
        type=int,
        default=256,
        help="Approximate memory budget per read/write block.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_dir = Path(args.src_dir)
    output_dir = Path(args.output_dir) if args.output_dir else src_dir / "tif"
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in ("beijing_label.mat", "wuhan_label.mat"):
        convert_label_file(src_dir / name, output_dir, args.block_mb)

    for name in ("beijing.mat", "wuhan.mat"):
        convert_scene_file(src_dir / name, output_dir, args.block_mb)


if __name__ == "__main__":
    main()
