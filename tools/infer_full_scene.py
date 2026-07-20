"""Infer C2Seg AB/BW full-scene images with trained patch models.

Examples:
    # Run one model on BW Wuhan, using default data root from .env
    python tools/infer_full_scene.py --dataset BW --scene wuhan \
        --model_name cxup_4b_BW_PMRG_v2_lossV2

    # Run every available AB checkpoint found under output/*_AB
    python tools/infer_full_scene.py --dataset AB --scene berlin --all

    # Use explicit full-scene TIFF paths
    python tools/infer_full_scene.py --config PaddleCD/c2seg_config/unet_BW.yml \
        --model_path output/unet_BW/best_model/model.pdparams \
        --msi path/to/beijing_MSI.tif --sar path/to/beijing_SAR.tif \
        --hsi path/to/beijing_HSI.tif
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tifffile


REPO_ROOT = Path(__file__).resolve().parents[1]
PADDLECD_ROOT = REPO_ROOT / "PaddleCD"
if str(PADDLECD_ROOT) not in sys.path:
    sys.path.insert(0, str(PADDLECD_ROOT))


CLASS_NAMES = [
    "Background",
    "Surface water",
    "Street",
    "Urban Fabric",
    "Industrial, commercial and transport",
    "Mine, dump, and construction sites",
    "Artificial, vegetated areas",
    "Arable Land",
    "Permanent Crops",
    "Pastures",
    "Forests",
    "Shrub",
    "Open spaces with no vegetation",
    "Inland wetlands",
]

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


@dataclass
class ModelSpec:
    name: str
    config: Path
    model_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("AB", "BW"), default="", help="C2Seg dataset.")
    parser.add_argument(
        "--scene",
        default="",
        help="Scene name. AB: berlin/augsburg. BW: beijing/wuhan.",
    )
    parser.add_argument(
        "--data_root",
        default="",
        help="C2Seg src root containing tif/ and C2Seg_AB/C2Seg_BW. Defaults to .env-derived root.",
    )
    parser.add_argument("--scene_root", default="", help="Optional directory containing one scene.")
    parser.add_argument("--msi", default="", help="Explicit full-scene MSI TIFF.")
    parser.add_argument("--sar", default="", help="Explicit full-scene SAR TIFF.")
    parser.add_argument("--hsi", default="", help="Explicit full-scene HSI TIFF.")
    parser.add_argument(
        "--mat",
        default="",
        help="Explicit official full-scene MATLAB v7.3 file containing MSI/SAR/HSI.",
    )
    parser.add_argument("--config", default="", help="Single PaddleCD config path.")
    parser.add_argument("--model_path", default="", help="Single model .pdparams path.")
    parser.add_argument(
        "--model_name",
        nargs="+",
        default=None,
        help="One or more model names. Uses PaddleCD/c2seg_config/<name>.yml and output/<name>/best_model/model.pdparams.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Infer all matching checkpoints found under output/*_<dataset>/best_model/model.pdparams.",
    )
    parser.add_argument(
        "--output_dir",
        default="ana/full_scene",
        help="Root output directory. Outputs go to <output_dir>/<dataset>_<scene>/<model_name>.",
    )
    parser.add_argument("--crop_size", nargs=2, type=int, default=[256, 256], metavar=("W", "H"))
    parser.add_argument("--stride", nargs=2, type=int, default=[256, 256], metavar=("W", "H"))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--msi_bands", nargs="+", type=int, default=None, help="1-based MSI bands.")
    parser.add_argument("--sar_bands", nargs="+", type=int, default=None, help="1-based SAR bands.")
    parser.add_argument("--hsi_bands", nargs="+", type=int, default=None, help="1-based HSI bands.")
    parser.add_argument("--hsi_bands_file", default="", help="Text file with 1-based HSI bands.")
    parser.add_argument("--rgb_bands", nargs=3, type=int, default=[1, 2, 3], metavar=("R", "G", "B"))
    parser.add_argument("--vis_downsample", type=int, default=0)
    parser.add_argument("--max_vis_pixels", type=int, default=25_000_000)
    parser.add_argument("--overlay_alpha", type=float, default=0.55)
    parser.add_argument("--keep_logits", action="store_true", help="Keep overlap logit memmaps.")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Load source/model and run one crop forward pass without writing outputs.",
    )
    return parser.parse_args()


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def env_data_root(dataset: str) -> Path | None:
    load_dotenv(REPO_ROOT / ".env")
    candidates = ["C2SEG_DATA_ROOT", "C2SEG_ROOT"]
    if dataset:
        candidates.append(f"C2SEG_{dataset}_ROOT")
    for key in candidates:
        value = os.environ.get(key)
        if not value:
            continue
        path = Path(value)
        if path.name.upper() in {"C2SEG_AB", "C2SEG_BW"}:
            return path.parent
        if path.name.lower() == "train" and path.parent.name.upper() in {"C2SEG_AB", "C2SEG_BW"}:
            return path.parent.parent
        return path
    return None


def import_paddle_runtime():
    import paddle

    from paddleseg.cvlibs import Config
    from paddleseg.utils import get_sys_env
    from paddleseg.utils import utils as seg_utils

    return paddle, Config, get_sys_env, seg_utils


def choose_device(requested: str, paddle, get_sys_env) -> str:
    if requested != "auto":
        return requested
    env_info = get_sys_env()
    if env_info["Paddle compiled with cuda"] and env_info["GPUs used"]:
        return "gpu"
    return "cpu"


def parse_bands_file(path: str) -> list[int] | None:
    if not path:
        return None
    text = Path(path).read_text(encoding="utf-8")
    return [int(token) for token in text.replace(",", " ").split()]


def channel_axis(shape: tuple[int, ...]) -> int:
    if len(shape) != 3:
        raise ValueError(f"Expected a 3D image stack, got shape={shape}")
    if shape[0] <= 512 and shape[1] > 512 and shape[2] > 512:
        return 0
    if shape[2] <= 512:
        return 2
    return 0


class ArrayReader:
    def __init__(self, array, name: str):
        self.array = array
        self.name = name
        self.shape = tuple(int(v) for v in array.shape)
        if len(self.shape) == 2:
            self.count = 1
            self.height, self.width = self.shape
            self.axis = None
        else:
            self.axis = channel_axis(self.shape)
            if self.axis == 0:
                self.count, self.height, self.width = self.shape
            else:
                self.height, self.width, self.count = self.shape

    def resolve_bands(self, requested: list[int] | None) -> list[int]:
        bands = list(range(1, self.count + 1)) if requested is None else list(requested)
        bad = [idx for idx in bands if idx < 1 or idx > self.count]
        if bad:
            raise ValueError(f"{self.name} bands {bad} outside valid range 1..{self.count}")
        return bands

    def read_chw(self, x: int, y: int, w: int, h: int, bands: list[int]) -> np.ndarray:
        idx = [b - 1 for b in bands]
        if self.axis == 0:
            block = np.asarray(self.array[:, y : y + h, x : x + w])
            out = block[idx, :, :]
        elif self.axis == 2:
            block = np.asarray(self.array[y : y + h, x : x + w, :])
            out = block[:, :, idx]
            out = np.transpose(out, (2, 0, 1))
        else:
            out = np.asarray(self.array[y : y + h, x : x + w])[None, :, :]
        return out.astype("float32", copy=False)

    def read_downsampled_chw(self, bands: list[int], step: int) -> np.ndarray:
        idx = [b - 1 for b in bands]
        if self.axis == 0:
            block = np.asarray(self.array[:, ::step, ::step])
            out = block[idx, :, :]
        elif self.axis == 2:
            block = np.asarray(self.array[::step, ::step, :])
            out = np.transpose(block[:, :, idx], (2, 0, 1))
        else:
            out = np.asarray(self.array[::step, ::step])[None, :, :]
        return out.astype("float32", copy=False)

    def close(self) -> None:
        pass


class TiffReader(ArrayReader):
    def __init__(self, path: Path, name: str):
        self.path = path
        try:
            array = tifffile.memmap(path)
        except ValueError:
            array = tifffile.imread(path)
        super().__init__(array, name)


class H5Reader(ArrayReader):
    def __init__(self, file_handle: h5py.File, dataset_key: str, name: str):
        self.file_handle = file_handle
        self.dataset_key = dataset_key
        super().__init__(file_handle[dataset_key], name)


class FullSceneSource:
    def __init__(self, msi: ArrayReader, sar: ArrayReader, hsi: ArrayReader, source_paths: dict[str, str]):
        self.msi = msi
        self.sar = sar
        self.hsi = hsi
        self.source_paths = source_paths
        self.width = msi.width
        self.height = msi.height
        if (sar.width, sar.height) != (self.width, self.height):
            raise ValueError(
                f"SAR shape {sar.width}x{sar.height} differs from MSI {self.width}x{self.height}"
            )

    def close(self) -> None:
        self.msi.close()
        self.sar.close()
        self.hsi.close()

    def read_patch(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        msi_bands: list[int],
        sar_bands: list[int],
        hsi_bands: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        msi = self.msi.read_chw(x, y, w, h, msi_bands)
        sar = self.sar.read_chw(x, y, w, h, sar_bands)

        hx0 = int(math.floor(x * self.hsi.width / self.width))
        hy0 = int(math.floor(y * self.hsi.height / self.height))
        hx1 = int(math.ceil((x + w) * self.hsi.width / self.width))
        hy1 = int(math.ceil((y + h) * self.hsi.height / self.height))
        hx1 = min(max(hx1, hx0 + 1), self.hsi.width)
        hy1 = min(max(hy1, hy0 + 1), self.hsi.height)
        hsi = self.hsi.read_chw(hx0, hy0, hx1 - hx0, hy1 - hy0, hsi_bands)
        if hsi.shape[1:] != (h, w):
            hsi = resize_chw(hsi, (h, w))
        return np.concatenate([msi, sar], axis=0), hsi


def resize_chw(arr: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    out_h, out_w = out_hw
    hwc = np.transpose(arr, (1, 2, 0))
    resized = cv2.resize(hwc, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    if resized.ndim == 2:
        resized = resized[:, :, None]
    return np.transpose(resized, (2, 0, 1)).astype("float32", copy=False)


def find_key(keys: Iterable[str], wanted: str) -> str | None:
    lowered = {key.lower(): key for key in keys}
    if wanted.lower() in lowered:
        return lowered[wanted.lower()]
    for key in keys:
        if wanted.lower() in key.lower():
            return key
    return None


def open_mat_source(path: Path) -> FullSceneSource:
    handle = h5py.File(path, "r")
    keys = list(handle.keys())
    msi_key = find_key(keys, "MSI")
    sar_key = find_key(keys, "SAR")
    hsi_key = find_key(keys, "HSI")
    if not (msi_key and sar_key and hsi_key):
        handle.close()
        raise ValueError(f"{path} does not contain MSI/SAR/HSI datasets. keys={keys}")
    source = FullSceneSource(
        H5Reader(handle, msi_key, "MSI"),
        H5Reader(handle, sar_key, "SAR"),
        H5Reader(handle, hsi_key, "HSI"),
        {"mat": str(path), "MSI": f"{path}:{msi_key}", "SAR": f"{path}:{sar_key}", "HSI": f"{path}:{hsi_key}"},
    )

    def close_with_handle() -> None:
        handle.close()

    source.close = close_with_handle  # type: ignore[method-assign]
    return source


def open_tiff_source(msi: Path, sar: Path, hsi: Path) -> FullSceneSource:
    return FullSceneSource(
        TiffReader(msi, "MSI"),
        TiffReader(sar, "SAR"),
        TiffReader(hsi, "HSI"),
        {"MSI": str(msi), "SAR": str(sar), "HSI": str(hsi)},
    )


def existing(path: Path) -> Path | None:
    return path if path.is_file() else None


def scene_candidates(data_root: Path, dataset: str, scene: str) -> tuple[list[Path], list[Path]]:
    roots = [
        data_root,
        data_root / "tif",
        data_root / f"tif_{dataset}",
        data_root / f"mat_{dataset}",
        data_root / f"C2Seg_{dataset}",
        data_root / f"C2Seg_{dataset}" / "tif",
        data_root / f"C2Seg_{dataset}" / f"tif_{dataset}",
    ]
    roots.extend([data_root / "full", data_root / f"C2Seg_{dataset}" / "full"])
    mat_names = [f"{scene}.mat", f"{scene}_multimodal.mat", f"{scene.lower()}.mat", f"{scene.lower()}_multimodal.mat"]
    tiff_roots = []
    mat_paths = []
    for root in roots:
        tiff_roots.append(root)
        for name in mat_names:
            mat_paths.append(root / name)
    return tiff_roots, mat_paths


def find_tiff_triplet(roots: list[Path], scene: str) -> tuple[Path, Path, Path] | None:
    scene_variants = {scene, scene.lower(), scene.capitalize()}
    modality_variants = {
        "MSI": ["MSI", "msi"],
        "SAR": ["SAR", "sar"],
        "HSI": ["HSI", "hsi"],
    }
    for root in roots:
        found = {}
        for modality, variants in modality_variants.items():
            candidates = []
            for scene_name in scene_variants:
                for mod in variants:
                    for suffix in (".tif", ".tiff", ".TIF", ".TIFF"):
                        candidates.extend(
                            [
                                root / f"{scene_name}_{mod}{suffix}",
                                root / scene_name / f"{mod}{suffix}",
                                root / scene_name / f"{scene_name}_{mod}{suffix}",
                            ]
                        )
            hit = next((path for path in candidates if path.is_file()), None)
            if hit:
                found[modality] = hit
        if set(found) == {"MSI", "SAR", "HSI"}:
            return found["MSI"], found["SAR"], found["HSI"]
    return None


def resolve_source(args: argparse.Namespace) -> FullSceneSource:
    if args.mat:
        return open_mat_source(Path(args.mat))
    if args.msi or args.sar or args.hsi:
        if not (args.msi and args.sar and args.hsi):
            raise ValueError("--msi, --sar, and --hsi must be provided together.")
        return open_tiff_source(Path(args.msi), Path(args.sar), Path(args.hsi))
    if args.scene_root:
        root = Path(args.scene_root)
        triplet = find_tiff_triplet([root], args.scene or root.name)
        if triplet:
            return open_tiff_source(*triplet)
        for name in (f"{args.scene}.mat", f"{args.scene}_multimodal.mat", f"{root.name}.mat"):
            path = root / name
            if path.is_file():
                return open_mat_source(path)

    if not args.dataset or not args.scene:
        raise ValueError("Pass --dataset and --scene, or explicit --mat/--msi/--sar/--hsi.")
    data_root = Path(args.data_root) if args.data_root else env_data_root(args.dataset)
    if data_root is None:
        raise ValueError("Could not infer data root. Pass --data_root or set C2SEG_AB_ROOT/C2SEG_BW_ROOT in .env.")

    tiff_roots, mat_paths = scene_candidates(data_root, args.dataset, args.scene)
    triplet = find_tiff_triplet(tiff_roots, args.scene)
    if triplet:
        return open_tiff_source(*triplet)
    mat_path = next((path for path in mat_paths if path.is_file()), None)
    if mat_path:
        return open_mat_source(mat_path)
    raise FileNotFoundError(
        f"Could not find full-scene TIFF triplet or MAT for dataset={args.dataset}, scene={args.scene}, root={data_root}"
    )


def get_normalize_params(cfg) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    normalize = None
    for item in cfg.dic["train_dataset"]["transforms"]:
        if item.get("type") == "Normalize2":
            normalize = item
            break
    if normalize is None:
        raise ValueError("Could not find Normalize2 in train_dataset.transforms.")
    mean1 = np.asarray(normalize["mean1"], dtype=np.float32)
    std1 = np.asarray(normalize["std1"], dtype=np.float32)
    mean2 = np.asarray(normalize["mean2"], dtype=np.float32)
    std2 = np.asarray(normalize["std2"], dtype=np.float32)
    if np.any(std1 == 0) or np.any(std2 == 0):
        raise ValueError("Normalize2 std contains zero.")
    return mean1, std1, mean2, std2


def axis_starts(length: int, crop: int, stride: int) -> list[int]:
    if crop <= 0 or stride <= 0:
        raise ValueError("crop_size and stride must be positive.")
    if length <= crop:
        return [0]
    starts = list(range(0, length - crop + 1, stride))
    last = length - crop
    if starts[-1] != last:
        starts.append(last)
    return starts


def iter_windows(width: int, height: int, crop_size: list[int], stride: list[int]):
    crop_w, crop_h = crop_size
    stride_w, stride_h = stride
    for y in axis_starts(height, crop_h, stride_h):
        for x in axis_starts(width, crop_w, stride_w):
            yield x, y, min(crop_w, width - x), min(crop_h, height - y)


def normalize_inputs(im1: np.ndarray, im2: np.ndarray, mean1, std1, mean2, std2) -> tuple[np.ndarray, np.ndarray]:
    im1 = (im1 - mean1[:, None, None]) / std1[:, None, None]
    im2 = (im2 - mean2[:, None, None]) / std2[:, None, None]
    return im1.astype("float32", copy=False), im2.astype("float32", copy=False)


def forward_logits(model, im1, im2, paddle):
    if hasattr(model, "data_format") and model.data_format == "NHWC":
        im1 = im1.transpose((0, 2, 3, 1))
        im2 = im2.transpose((0, 2, 3, 1))
    logits = model(im1, im2)
    if len(logits) == 1:
        logit = logits[0]
    elif len(logits) == 2:
        logit = logits[0] + logits[1] * 0.4
    else:
        logit = logits[0]
    if len(logit.shape) == 3:
        logit = logit.unsqueeze(1)
    if hasattr(model, "data_format") and model.data_format == "NHWC":
        logit = logit.transpose((0, 3, 1, 2))
    return logit


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


def color_map(num_classes: int) -> np.ndarray:
    colors = np.zeros((num_classes, 3), dtype=np.uint8)
    for idx in range(num_classes):
        colors[idx] = BRIGHT_COLORS[idx] if idx < len(BRIGHT_COLORS) else hsv_to_rgb_uint8((idx * 47) % 360, 0.82, 1.0)
    return colors


def create_tiff_memmap(path: Path, shape: tuple[int, ...], dtype: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    bytes_total = math.prod(shape) * np.dtype(dtype).itemsize
    return tifffile.memmap(path, shape=shape, dtype=dtype, bigtiff=bytes_total > 3_800_000_000)


def stretch_band(band: np.ndarray) -> np.ndarray:
    finite = np.isfinite(band)
    if not np.any(finite):
        return np.zeros_like(band, dtype=np.uint8)
    valid = band[finite]
    lo, hi = np.percentile(valid, [2, 98])
    if hi <= lo:
        hi = lo + 1
    out = np.clip((band - lo) / (hi - lo), 0, 1)
    return (out * 255).astype(np.uint8)


def preview_rgb(source: FullSceneSource, rgb_bands: list[int], downsample: int) -> np.ndarray:
    bands = source.msi.resolve_bands(rgb_bands)
    raw = source.msi.read_downsampled_chw(bands, max(1, downsample))
    return np.stack([stretch_band(raw[idx]) for idx in range(3)], axis=-1)


def save_visuals(
    output_dir: Path,
    pred_path: Path,
    source: FullSceneSource,
    num_classes: int,
    args: argparse.Namespace,
) -> dict[str, str]:
    pred = tifffile.memmap(pred_path)
    colors = color_map(num_classes)
    color_path = output_dir / "prediction_color.tif"
    color = create_tiff_memmap(color_path, (source.height, source.width, 3), "uint8")
    block = 1024
    for y in range(0, source.height, block):
        end = min(y + block, source.height)
        color[y:end, :, :] = colors[np.asarray(pred[y:end, :], dtype=np.int64)]
    color.flush()
    del color

    downsample = args.vis_downsample
    if downsample <= 0:
        downsample = max(1, int(math.ceil(math.sqrt((source.width * source.height) / args.max_vis_pixels))))
    pred_preview = np.asarray(pred[::downsample, ::downsample])
    color_preview = colors[pred_preview]
    rgb = preview_rgb(source, args.rgb_bands, downsample)
    min_h = min(rgb.shape[0], color_preview.shape[0])
    min_w = min(rgb.shape[1], color_preview.shape[1])
    rgb = rgb[:min_h, :min_w]
    color_preview = color_preview[:min_h, :min_w]
    pred_preview = pred_preview[:min_h, :min_w]
    overlay = (
        rgb.astype("float32") * args.overlay_alpha
        + color_preview.astype("float32") * (1.0 - args.overlay_alpha)
    ).round().clip(0, 255).astype("uint8")

    gray_preview_path = output_dir / "prediction_gray_preview.png"
    color_preview_path = output_dir / "prediction_color_preview.png"
    rgb_path = output_dir / "msi_rgb_preview.png"
    overlay_path = output_dir / "overlay_preview.png"
    Image.fromarray(pred_preview).save(gray_preview_path)
    Image.fromarray(color_preview).save(color_preview_path)
    Image.fromarray(rgb).save(rgb_path)
    Image.fromarray(overlay).save(overlay_path)
    save_legend(output_dir / "legend.png", colors)
    del pred
    return {
        "prediction_color_tif": str(color_path),
        "prediction_gray_preview": str(gray_preview_path),
        "prediction_color_preview": str(color_preview_path),
        "msi_rgb_preview": str(rgb_path),
        "overlay_preview": str(overlay_path),
        "legend": str(output_dir / "legend.png"),
    }


def save_legend(path: Path, colors: np.ndarray) -> None:
    names = CLASS_NAMES[: len(colors)]
    if len(names) < len(colors):
        names.extend([f"Class {idx}" for idx in range(len(names), len(colors))])
    swatch = 24
    pad = 12
    row_h = 34
    image = Image.new("RGB", (520, pad * 2 + row_h * len(names)), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for idx, name in enumerate(names):
        y = pad + idx * row_h
        draw.rectangle([pad, y + 4, pad + swatch, y + 4 + swatch], fill=tuple(colors[idx]), outline=(60, 60, 60))
        draw.text((pad + swatch + 12, y + 8), f"{idx}: {name}", fill=(20, 20, 20), font=font)
    image.save(path)


def resolve_model_specs(args: argparse.Namespace) -> list[ModelSpec]:
    if args.config or args.model_path:
        if not (args.config and args.model_path):
            raise ValueError("--config and --model_path must be provided together.")
        name = Path(args.config).stem
        return [ModelSpec(name=name, config=Path(args.config), model_path=Path(args.model_path))]

    names = args.model_name or []
    if args.all:
        if not args.dataset:
            raise ValueError("--all requires --dataset AB or --dataset BW.")
        suffix = f"_{args.dataset}"
        for path in sorted((REPO_ROOT / "output").glob(f"*{suffix}*/best_model/model.pdparams")):
            names.append(path.parents[1].name)
    if not names:
        raise ValueError("Pass --model_name, --all, or --config plus --model_path.")

    specs = []
    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        config = REPO_ROOT / "PaddleCD" / "c2seg_config" / f"{name}.yml"
        model_path = REPO_ROOT / "output" / name / "best_model" / "model.pdparams"
        if not config.is_file():
            if args.all and args.model_name is None:
                print(f"Skip {name}: missing config {config}", flush=True)
                continue
            raise FileNotFoundError(f"Missing config for {name}: {config}")
        if not model_path.is_file():
            if args.all and args.model_name is None:
                print(f"Skip {name}: missing checkpoint {model_path}", flush=True)
                continue
            raise FileNotFoundError(f"Missing checkpoint for {name}: {model_path}")
        specs.append(ModelSpec(name=name, config=config, model_path=model_path))
    if not specs:
        raise ValueError("No runnable model specs found.")
    return specs


def num_classes_from_cfg(cfg) -> int:
    return int(cfg.dic.get("model", {}).get("num_classes", cfg.dic["train_dataset"].get("num_classes", 14)))


def run_one_model(spec: ModelSpec, source: FullSceneSource, args: argparse.Namespace, runtime) -> Path:
    paddle, Config, _get_sys_env, seg_utils = runtime
    cfg = Config(str(spec.config))
    model = cfg.model
    seg_utils.load_entire_model(model, str(spec.model_path))
    model.eval()

    mean1, std1, mean2, std2 = get_normalize_params(cfg)
    msi_bands = source.msi.resolve_bands(args.msi_bands)
    sar_bands = source.sar.resolve_bands(args.sar_bands)
    hsi_requested = args.hsi_bands or parse_bands_file(args.hsi_bands_file)
    if hsi_requested is None and source.hsi.count > len(mean2):
        hsi_requested = list(range(1, len(mean2) + 1))
        print(
            f"[{spec.name}] HSI has {source.hsi.count} bands, config expects {len(mean2)}; "
            f"using bands 1..{len(mean2)}. Pass --hsi_bands to override.",
            flush=True,
        )
    hsi_bands = source.hsi.resolve_bands(hsi_requested)
    if len(msi_bands) + len(sar_bands) != len(mean1):
        raise ValueError(f"{spec.name}: MSI+SAR channels {len(msi_bands)+len(sar_bands)} != mean1 length {len(mean1)}")
    if len(hsi_bands) != len(mean2):
        raise ValueError(f"{spec.name}: HSI channels {len(hsi_bands)} != mean2 length {len(mean2)}")

    if args.dry_run:
        paddle, _Config, _get_sys_env, _seg_utils = runtime
        x, y, w, h = next(iter_windows(source.width, source.height, args.crop_size, args.stride))
        im1, im2 = source.read_patch(x, y, w, h, msi_bands, sar_bands, hsi_bands)
        im1, im2 = normalize_inputs(im1, im2, mean1, std1, mean2, std2)
        with paddle.no_grad():
            logits = forward_logits(
                model,
                paddle.to_tensor(im1[None, :, :, :]),
                paddle.to_tensor(im2[None, :, :, :]),
                paddle,
            )
        print(
            f"[{spec.name}] dry_run ok: im1={im1.shape}, im2={im2.shape}, logits={tuple(logits.shape)}",
            flush=True,
        )
        del model
        return Path(".")

    scene_tag = "_".join(part for part in [args.dataset, args.scene] if part) or "custom_scene"
    output_dir = Path(args.output_dir) / scene_tag / spec.name
    output_dir.mkdir(parents=True, exist_ok=True)

    num_classes = num_classes_from_cfg(cfg)
    pred_path = output_dir / "prediction_gray.tif"
    pred = create_tiff_memmap(pred_path, (source.height, source.width), "uint8")
    windows = list(iter_windows(source.width, source.height, args.crop_size, args.stride))
    no_overlap = args.crop_size == args.stride
    logit_sum = None
    count = None
    if not no_overlap:
        logit_sum_path = output_dir / "logit_sum.dat"
        count_path = output_dir / "count.dat"
        logit_sum = np.memmap(logit_sum_path, mode="w+", dtype="float32", shape=(num_classes, source.height, source.width))
        count = np.memmap(count_path, mode="w+", dtype="uint16", shape=(source.height, source.width))
        logit_sum[:] = 0
        count[:] = 0

    print(f"[{spec.name}] windows={len(windows)} size={source.width}x{source.height} output={output_dir}", flush=True)
    batch1, batch2, batch_windows = [], [], []
    processed = 0
    with paddle.no_grad():
        for x, y, w, h in windows:
            im1, im2 = source.read_patch(x, y, w, h, msi_bands, sar_bands, hsi_bands)
            im1, im2 = normalize_inputs(im1, im2, mean1, std1, mean2, std2)
            batch1.append(im1)
            batch2.append(im2)
            batch_windows.append((x, y, w, h))
            if len(batch1) == args.batch_size or processed + len(batch1) == len(windows):
                logits = forward_logits(
                    model,
                    paddle.to_tensor(np.stack(batch1, axis=0)),
                    paddle.to_tensor(np.stack(batch2, axis=0)),
                    paddle,
                ).numpy()
                if no_overlap:
                    patch_preds = np.argmax(logits, axis=1).astype("uint8")
                    for patch_pred, (px, py, pw, ph) in zip(patch_preds, batch_windows):
                        pred[py : py + ph, px : px + pw] = patch_pred[:ph, :pw]
                else:
                    assert logit_sum is not None and count is not None
                    for patch_logit, (px, py, pw, ph) in zip(logits, batch_windows):
                        logit_sum[:, py : py + ph, px : px + pw] += patch_logit[:, :ph, :pw]
                        count[py : py + ph, px : px + pw] += 1
                processed += len(batch1)
                print(f"[{spec.name}] {processed}/{len(windows)}", flush=True)
                batch1.clear()
                batch2.clear()
                batch_windows.clear()

    if not no_overlap:
        assert logit_sum is not None and count is not None
        if np.any(count == 0):
            raise RuntimeError("Some pixels were not covered by sliding windows.")
        block = 512
        for y in range(0, source.height, block):
            end = min(y + block, source.height)
            pred[y:end, :] = np.argmax(logit_sum[:, y:end, :] / count[y:end, :][None, :, :], axis=0).astype("uint8")
        logit_sum.flush()
        count.flush()
        del logit_sum
        del count
        if not args.keep_logits:
            for name in ("logit_sum.dat", "count.dat"):
                try:
                    (output_dir / name).unlink()
                except OSError:
                    pass

    pred.flush()
    del pred
    visual_outputs = save_visuals(output_dir, pred_path, source, num_classes, args)
    metadata = {
        "model_name": spec.name,
        "config": str(spec.config),
        "model_path": str(spec.model_path),
        "dataset": args.dataset,
        "scene": args.scene,
        "source": source.source_paths,
        "width": source.width,
        "height": source.height,
        "num_classes": num_classes,
        "crop_size": args.crop_size,
        "stride": args.stride,
        "batch_size": args.batch_size,
        "msi_bands": msi_bands,
        "sar_bands": sar_bands,
        "hsi_bands": hsi_bands,
        "outputs": {"prediction_gray_tif": str(pred_path), **visual_outputs},
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    del model
    gc.collect()
    if args.device == "gpu" or args.device == "auto":
        try:
            paddle.device.cuda.empty_cache()
        except Exception:
            pass
    return output_dir


def main() -> None:
    args = parse_args()
    specs = resolve_model_specs(args)
    paddle, Config, get_sys_env, seg_utils = import_paddle_runtime()
    device = choose_device(args.device, paddle, get_sys_env)
    paddle.set_device(device)
    runtime = (paddle, Config, get_sys_env, seg_utils)
    source = resolve_source(args)
    print("Source:", source.source_paths)
    print(f"Source size: MSI/SAR={source.width}x{source.height}, HSI={source.hsi.width}x{source.hsi.height}")
    print("Device:", device)
    print("Models:", ", ".join(spec.name for spec in specs))
    try:
        outputs = [run_one_model(spec, source, args, runtime) for spec in specs]
    finally:
        source.close()
    print("Done:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
