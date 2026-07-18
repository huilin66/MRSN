"""Create preview thumbnails for C2Seg full-scene TIFF files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
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


def stride_for_shape(height: int, width: int, max_dim: int) -> int:
    return max(1, int(np.ceil(max(height, width) / max_dim)))


def scale_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype("float32", copy=False)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(arr[valid], [2, 98])
    if hi <= lo:
        hi = lo + 1
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (arr * 255).astype(np.uint8)


def choose_bands(arr: np.ndarray, name: str) -> list[int]:
    bands = arr.shape[0]
    lower = name.lower()
    if "msi" in lower:
        return list(range(min(3, bands)))
    if "sar" in lower:
        return [0, min(1, bands - 1), 0]
    if "hsi" in lower:
        return [min(60, bands - 1), min(35, bands - 1), min(15, bands - 1)]
    return [0, min(1, bands - 1), min(2, bands - 1)]


def thumbnail_array(path: Path, max_dim: int) -> np.ndarray:
    arr = tifffile.memmap(path)
    shape = arr.shape
    name = path.name.lower()

    if arr.ndim == 2:
        step = stride_for_shape(shape[0], shape[1], max_dim)
        sample = np.asarray(arr[::step, ::step])
        if "label" in name:
            palette = build_palette(256)
            color_index = np.where(sample < 0, 0, sample).astype(np.int64) % 256
            return palette[color_index]
        return scale_to_uint8(sample)

    if arr.ndim == 3 and shape[-1] in (3, 4):
        step = stride_for_shape(shape[0], shape[1], max_dim)
        sample = np.asarray(arr[::step, ::step, :3])
        if sample.dtype == np.uint8:
            return sample
        channels = [scale_to_uint8(sample[:, :, idx]) for idx in range(3)]
        return np.stack(channels, axis=-1)

    if arr.ndim == 3:
        _, height, width = shape
        step = stride_for_shape(height, width, max_dim)
        channels = []
        for band in choose_bands(arr, name):
            sample = np.asarray(arr[band, ::step, ::step])
            channels.append(scale_to_uint8(sample))
        while len(channels) < 3:
            channels.append(channels[-1])
        return np.stack(channels[:3], axis=-1)

    raise ValueError(f"Unsupported TIFF shape for {path}: {shape}")


def save_thumbnail(path: Path, output_dir: Path, max_dim: int) -> Path:
    image = thumbnail_array(path, max_dim)
    output_path = output_dir / f"{path.stem}_thumb.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output_path)
    print(f"{path.name} -> {output_path.name} {image.shape}", flush=True)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tif_dir", required=True, help="Directory containing TIFF files.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Thumbnail output directory. Defaults to <tif_dir>/thumbnails.",
    )
    parser.add_argument("--max_dim", type=int, default=1600, help="Longest thumbnail edge.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tif_dir = Path(args.tif_dir)
    output_dir = Path(args.output_dir) if args.output_dir else tif_dir / "thumbnails"
    for path in sorted(tif_dir.glob("*.tif")):
        save_thumbnail(path, output_dir, args.max_dim)


if __name__ == "__main__":
    main()
