"""Infer a trained patch model on one full-size C2Seg scene.

The patch training configs in PaddleCD use RS_MD3B inputs:
    t1 = concat(MSI, SAR), shape [6, H, W]
    t2 = HSI, shape [hsi_chs, H, W]

This tool reads original full-size MSI/SAR/HSI GeoTIFFs, runs sliding-window
inference, stitches class probabilities back to the full image, and writes
analysis-friendly visualizations.

Example:
    python tools/infer_full_scene.py \
        --config PaddleCD/c2seg_config/cxup_4b_BW_PMRG_v2_lossV2.yml \
        --model_path output/cxup_4b_BW_PMRG_v2_lossV2/best_model/model.pdparams \
        --msi /path/to/original_BW/msi.tif \
        --sar /path/to/original_BW/sar.tif \
        --hsi /path/to/original_BW/hsi.tif \
        --output_dir ana/full_scene/cxup_4b_BW_PMRG_v2_lossV2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


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

BRIGHT_COLORS = np.array(
    [
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
    ],
    dtype=np.uint8,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run patch-trained C2Seg models on an original full-size scene."
    )
    parser.add_argument("--config", dest="cfg", required=True, help="PaddleCD YAML config.")
    parser.add_argument("--model_path", required=True, help="Trained model .pdparams path.")
    parser.add_argument(
        "--scene_root",
        default="",
        help="Optional directory containing msi/sar/hsi tif files. Explicit paths win.",
    )
    parser.add_argument("--msi", default="", help="Original full-size MSI GeoTIFF.")
    parser.add_argument("--sar", default="", help="Original full-size SAR GeoTIFF.")
    parser.add_argument("--hsi", default="", help="Original full-size HSI GeoTIFF.")
    parser.add_argument(
        "--msi_bands",
        nargs="+",
        type=int,
        default=None,
        help="Optional 1-based MSI band indexes to read. Default: all bands.",
    )
    parser.add_argument(
        "--sar_bands",
        nargs="+",
        type=int,
        default=None,
        help="Optional 1-based SAR band indexes to read. Default: all bands.",
    )
    parser.add_argument(
        "--hsi_bands",
        nargs="+",
        type=int,
        default=None,
        help="Optional 1-based HSI band indexes to read. Default: all bands.",
    )
    parser.add_argument(
        "--hsi_bands_file",
        default="",
        help="Optional text file with 1-based HSI band indexes, separated by spaces, commas, or newlines.",
    )
    parser.add_argument(
        "--output_dir",
        default="ana/full_scene",
        help="Output directory. Default: ana/full_scene",
    )
    parser.add_argument(
        "--crop_size",
        nargs=2,
        type=int,
        default=[256, 256],
        metavar=("WIDTH", "HEIGHT"),
        help="Sliding crop size in pixels. Default: 256 256",
    )
    parser.add_argument(
        "--stride",
        nargs=2,
        type=int,
        default=[256, 256],
        metavar=("WIDTH", "HEIGHT"),
        help="Sliding stride in pixels. Default: 256 256",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Patch batch size.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Inference device. Default: auto",
    )
    parser.add_argument(
        "--data_format",
        choices=("NCHW", "NHWC"),
        default="NCHW",
        help="Same constraint as PaddleCD/val.py. Default: NCHW",
    )
    parser.add_argument(
        "--rgb_bands",
        nargs=3,
        type=int,
        default=[1, 2, 3],
        metavar=("R", "G", "B"),
        help="1-based MSI bands for RGB overlay. Default: 1 2 3",
    )
    parser.add_argument(
        "--vis_downsample",
        type=int,
        default=0,
        help="PNG visualization downsample. 0 chooses automatically. Default: 0",
    )
    parser.add_argument(
        "--max_vis_pixels",
        type=int,
        default=25_000_000,
        help="Max pixels for PNG previews when --vis_downsample=0. Default: 25000000",
    )
    parser.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.55,
        help="RGB opacity in overlay; prediction opacity is 1-alpha. Default: 0.55",
    )
    parser.add_argument(
        "--keep_prob_memmap",
        action="store_true",
        help="Keep stitched probability-sum memmap for later inspection.",
    )
    return parser.parse_args()


def import_runtime():
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.windows import Window
    except ImportError as exc:
        raise RuntimeError(
            "This script needs rasterio for full-scene GeoTIFF window reads. "
            "Install it in the Paddle environment before running."
        ) from exc

    import paddle
    import paddle.nn.functional as F

    from paddleseg.cvlibs import Config
    from paddleseg.utils import get_sys_env
    from paddleseg.utils import utils as seg_utils

    return rasterio, Resampling, Window, paddle, F, Config, get_sys_env, seg_utils


def choose_device(requested, paddle, get_sys_env):
    if requested != "auto":
        return requested
    env_info = get_sys_env()
    if env_info["Paddle compiled with cuda"] and env_info["GPUs used"]:
        return "gpu"
    return "cpu"


def apply_data_format(cfg, data_format):
    if data_format != "NHWC":
        return
    if cfg.dic["model"]["type"] != "DeepLabV3P":
        raise ValueError('The "NHWC" data format only supports DeepLabV3P.')
    cfg.dic["model"]["data_format"] = data_format
    cfg.dic["model"]["backbone"]["data_format"] = data_format
    for loss_cfg in cfg.dic["loss"]["types"]:
        loss_cfg["data_format"] = data_format


def find_scene_file(scene_root, modality):
    if not scene_root:
        return ""
    root = Path(scene_root)
    candidates = []
    for suffix in (".tif", ".tiff", ".TIF", ".TIFF"):
        candidates.extend(
            [
                root / f"{modality}{suffix}",
                root / f"{modality.upper()}{suffix}",
                root / modality / f"{modality}{suffix}",
                root / modality.upper() / f"{modality.upper()}{suffix}",
            ]
        )
    for path in candidates:
        if path.is_file():
            return str(path)
    matches = []
    for suffix in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        matches.extend(root.rglob(suffix))
    matches = [p for p in matches if modality.lower() in p.stem.lower()]
    if len(matches) == 1:
        return str(matches[0])
    return ""


def resolve_scene_paths(args):
    msi = args.msi or find_scene_file(args.scene_root, "msi")
    sar = args.sar or find_scene_file(args.scene_root, "sar")
    hsi = args.hsi or find_scene_file(args.scene_root, "hsi")
    missing = [name for name, path in (("msi", msi), ("sar", sar), ("hsi", hsi)) if not path]
    if missing:
        raise ValueError(
            "Missing full-scene paths for {}. Pass --msi/--sar/--hsi explicitly.".format(
                ", ".join(missing)
            )
        )
    for path in (msi, sar, hsi):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    return msi, sar, hsi


def get_normalize_params(cfg):
    transforms = cfg.dic["train_dataset"]["transforms"]
    normalize = None
    for item in transforms:
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


def parse_bands_file(path):
    if not path:
        return None
    text = Path(path).read_text(encoding="utf-8")
    values = []
    for token in text.replace(",", " ").split():
        values.append(int(token))
    return values


def resolve_band_indexes(src, requested, name):
    if requested is None:
        return list(range(1, src.count + 1))
    if not requested:
        raise ValueError("{} band list is empty.".format(name))
    bad = [idx for idx in requested if idx < 1 or idx > src.count]
    if bad:
        raise ValueError(
            "{} band indexes {} are outside valid range 1..{}.".format(
                name, bad, src.count
            )
        )
    return list(requested)


def validate_sources(msi_src, sar_src, hsi_src, mean1, mean2, msi_bands, sar_bands, hsi_bands):
    shape = (msi_src.height, msi_src.width)
    for name, src in (("SAR", sar_src), ("HSI", hsi_src)):
        if (src.height, src.width) != shape:
            raise ValueError(
                "{} shape {}x{} differs from MSI shape {}x{}.".format(
                    name, src.width, src.height, msi_src.width, msi_src.height
                )
            )
    if len(msi_bands) + len(sar_bands) != len(mean1):
        raise ValueError(
            "Selected MSI+SAR channels ({}) do not match Normalize2 mean1 length ({}).".format(
                len(msi_bands) + len(sar_bands), len(mean1)
            )
        )
    if len(hsi_bands) != len(mean2):
        raise ValueError(
            "Selected HSI channels ({}) do not match Normalize2 mean2 length ({}). "
            "Use --hsi_bands or --hsi_bands_file if the original scene has extra bands.".format(
                len(hsi_bands), len(mean2)
            )
        )
    return shape


def axis_starts(length, crop, stride):
    if crop <= 0 or stride <= 0:
        raise ValueError("crop_size and stride must be positive.")
    if length <= crop:
        return [0]
    starts = list(range(0, length - crop + 1, stride))
    last = length - crop
    if starts[-1] != last:
        starts.append(last)
    return starts


def iter_windows(width, height, crop_size, stride):
    crop_w, crop_h = crop_size
    stride_w, stride_h = stride
    ys = axis_starts(height, crop_h, stride_h)
    xs = axis_starts(width, crop_w, stride_w)
    for y in ys:
        for x in xs:
            yield x, y, min(crop_w, width - x), min(crop_h, height - y)


def read_patch(src, window, Window, indexes, fill_value=0.0):
    data = src.read(indexes=indexes, window=window, boundless=True, fill_value=fill_value)
    return np.transpose(data, (1, 2, 0)).astype(np.float32, copy=False)


def normalize_patch(msi, sar, hsi, mean1, std1, mean2, std2):
    im1 = np.concatenate([msi, sar], axis=2)
    im1 = (im1 - mean1.reshape(1, 1, -1)) / std1.reshape(1, 1, -1)
    im2 = (hsi - mean2.reshape(1, 1, -1)) / std2.reshape(1, 1, -1)
    im1 = np.transpose(im1, (2, 0, 1))
    im2 = np.transpose(im2, (2, 0, 1))
    return im1, im2


def model_logits(model, im1, im2, paddle, F):
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
    return F.softmax(logit, axis=1)


def class_color_map(num_classes):
    if num_classes <= len(BRIGHT_COLORS):
        return BRIGHT_COLORS[:num_classes]
    colors = np.zeros((num_classes, 3), dtype=np.uint8)
    colors[: len(BRIGHT_COLORS)] = BRIGHT_COLORS
    for idx in range(len(BRIGHT_COLORS), num_classes):
        hue = (idx * 47) % 360
        colors[idx] = hsv_to_rgb_uint8(hue, 0.82, 1.0)
    return colors


def hsv_to_rgb_uint8(hue, saturation, value):
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
    return np.array([(r + m) * 255, (g + m) * 255, (b + m) * 255], dtype=np.uint8)


def downsample_nearest(arr, factor):
    if factor <= 1:
        return arr
    return arr[::factor, ::factor]


def choose_vis_downsample(width, height, requested, max_pixels):
    if requested and requested > 0:
        return requested
    if width * height <= max_pixels:
        return 1
    return int(math.ceil(math.sqrt((width * height) / float(max_pixels))))


def stretch_band(band, pmin=2, pmax=98):
    finite = np.isfinite(band)
    if not np.any(finite):
        return np.zeros_like(band, dtype=np.float32)
    valid = band[finite]
    lo, hi = np.percentile(valid, (pmin, pmax))
    if lo == hi:
        lo, hi = float(valid.min()), float(valid.max())
    if lo == hi:
        return np.zeros_like(band, dtype=np.float32)
    return np.clip((band - lo) / (hi - lo), 0, 1).astype(np.float32)


def read_msi_rgb(msi_src, rgb_bands, downsample, Resampling):
    if any(b < 1 or b > msi_src.count for b in rgb_bands):
        raise ValueError(
            "--rgb_bands {} are outside MSI band range 1..{}.".format(
                rgb_bands, msi_src.count
            )
        )
    out_h = max(1, int(math.ceil(msi_src.height / downsample)))
    out_w = max(1, int(math.ceil(msi_src.width / downsample)))
    data = msi_src.read(
        indexes=rgb_bands,
        out_shape=(3, out_h, out_w),
        resampling=Resampling.average,
    )
    channels = [stretch_band(data[i]) for i in range(3)]
    rgb = np.stack(channels, axis=2)
    return (rgb * 255).round().astype(np.uint8)


def save_png(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def save_legend(path, colors, class_names):
    swatch = 24
    pad = 12
    row_h = 34
    width = 520
    height = pad * 2 + row_h * len(class_names)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for idx, name in enumerate(class_names):
        y = pad + idx * row_h
        color = tuple(int(v) for v in colors[idx])
        draw.rectangle([pad, y + 4, pad + swatch, y + 4 + swatch], fill=color, outline=(60, 60, 60))
        draw.text((pad + swatch + 12, y + 8), f"{idx}: {name}", fill=(20, 20, 20), font=font)
    image.save(path)


def save_geotiff_with_rasterio(rasterio, path, pred, reference_src):
    profile = reference_src.profile.copy()
    profile.update(count=1, dtype="uint8", nodata=None, compress="lzw")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(pred.astype(np.uint8), 1)


def main():
    args = parse_args()
    (
        rasterio,
        Resampling,
        Window,
        paddle,
        F,
        Config,
        get_sys_env,
        seg_utils,
    ) = import_runtime()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device, paddle, get_sys_env)
    paddle.set_device(device)

    cfg = Config(args.cfg)
    apply_data_format(cfg, args.data_format)
    model = cfg.model
    seg_utils.load_entire_model(model, args.model_path)
    model.eval()

    mean1, std1, mean2, std2 = get_normalize_params(cfg)
    num_classes = int(cfg.dic["model"].get("num_classes", cfg.dic["train_dataset"].get("num_classes", 14)))
    colors = class_color_map(num_classes)
    class_names = CLASS_NAMES[:num_classes]
    if len(class_names) < num_classes:
        class_names.extend([f"Class {idx}" for idx in range(len(class_names), num_classes)])

    msi_path, sar_path, hsi_path = resolve_scene_paths(args)
    print("MSI:", msi_path)
    print("SAR:", sar_path)
    print("HSI:", hsi_path)
    print("Output:", output_dir)

    with rasterio.open(msi_path) as msi_src, rasterio.open(sar_path) as sar_src, rasterio.open(hsi_path) as hsi_src:
        hsi_requested_bands = args.hsi_bands or parse_bands_file(args.hsi_bands_file)
        msi_bands = resolve_band_indexes(msi_src, args.msi_bands, "MSI")
        sar_bands = resolve_band_indexes(sar_src, args.sar_bands, "SAR")
        hsi_bands = resolve_band_indexes(hsi_src, hsi_requested_bands, "HSI")
        height, width = validate_sources(
            msi_src, sar_src, hsi_src, mean1, mean2, msi_bands, sar_bands, hsi_bands
        )
        windows = list(iter_windows(width, height, args.crop_size, args.stride))
        print("Scene size: {}x{}, windows: {}, classes: {}".format(width, height, len(windows), num_classes))
        print("Bands: MSI {}, SAR {}, HSI {}".format(len(msi_bands), len(sar_bands), len(hsi_bands)))

        prob_path = output_dir / "prob_sum.dat"
        count_path = output_dir / "count.dat"
        prob_sum = np.memmap(prob_path, mode="w+", dtype="float32", shape=(num_classes, height, width))
        count = np.memmap(count_path, mode="w+", dtype="uint16", shape=(height, width))
        prob_sum[:] = 0
        count[:] = 0

        batch_im1 = []
        batch_im2 = []
        batch_windows = []
        processed = 0

        with paddle.no_grad():
            for x, y, w, h in windows:
                window = Window(x, y, w, h)
                msi = read_patch(msi_src, window, Window, msi_bands)
                sar = read_patch(sar_src, window, Window, sar_bands)
                hsi = read_patch(hsi_src, window, Window, hsi_bands)
                im1, im2 = normalize_patch(msi, sar, hsi, mean1, std1, mean2, std2)
                batch_im1.append(im1)
                batch_im2.append(im2)
                batch_windows.append((x, y, w, h))

                if len(batch_im1) == args.batch_size or processed + len(batch_im1) == len(windows):
                    tensor1 = paddle.to_tensor(np.stack(batch_im1, axis=0))
                    tensor2 = paddle.to_tensor(np.stack(batch_im2, axis=0))
                    probs = model_logits(model, tensor1, tensor2, paddle, F).numpy()

                    for patch_probs, (px, py, pw, ph) in zip(probs, batch_windows):
                        prob_sum[:, py : py + ph, px : px + pw] += patch_probs[:, :ph, :pw]
                        count[py : py + ph, px : px + pw] += 1

                    processed += len(batch_im1)
                    print("Processed {}/{} windows".format(processed, len(windows)), flush=True)
                    batch_im1.clear()
                    batch_im2.clear()
                    batch_windows.clear()

        if np.any(count == 0):
            raise RuntimeError("Some pixels were not covered by sliding windows.")

        pred = np.asarray(np.argmax(prob_sum, axis=0), dtype=np.uint8)
        pred_tif = output_dir / "prediction_gray.tif"
        save_geotiff_with_rasterio(rasterio, pred_tif, pred, msi_src)

        vis_downsample = choose_vis_downsample(width, height, args.vis_downsample, args.max_vis_pixels)
        pred_preview = downsample_nearest(pred, vis_downsample)
        color_preview = colors[pred_preview]
        rgb_preview = read_msi_rgb(msi_src, args.rgb_bands, vis_downsample, Resampling)
        if rgb_preview.shape[:2] != color_preview.shape[:2]:
            min_h = min(rgb_preview.shape[0], color_preview.shape[0])
            min_w = min(rgb_preview.shape[1], color_preview.shape[1])
            rgb_preview = rgb_preview[:min_h, :min_w]
            color_preview = color_preview[:min_h, :min_w]
            pred_preview = pred_preview[:min_h, :min_w]

        overlay = (
            rgb_preview.astype(np.float32) * args.overlay_alpha
            + color_preview.astype(np.float32) * (1.0 - args.overlay_alpha)
        ).round().clip(0, 255).astype(np.uint8)

        save_png(output_dir / "prediction_gray_preview.png", pred_preview)
        save_png(output_dir / "prediction_color.png", color_preview)
        save_png(output_dir / "msi_rgb.png", rgb_preview)
        save_png(output_dir / "overlay.png", overlay)
        save_legend(output_dir / "legend.png", colors, class_names)

        meta = {
            "config": os.path.abspath(args.cfg),
            "model_path": os.path.abspath(args.model_path),
            "msi": os.path.abspath(msi_path),
            "sar": os.path.abspath(sar_path),
            "hsi": os.path.abspath(hsi_path),
            "width": width,
            "height": height,
            "num_classes": num_classes,
            "crop_size": args.crop_size,
            "stride": args.stride,
            "batch_size": args.batch_size,
            "msi_bands": msi_bands,
            "sar_bands": sar_bands,
            "hsi_bands": hsi_bands,
            "windows": len(windows),
            "device": device,
            "vis_downsample": vis_downsample,
            "outputs": {
                "prediction_gray_tif": str(pred_tif),
                "prediction_gray_preview": str(output_dir / "prediction_gray_preview.png"),
                "prediction_color": str(output_dir / "prediction_color.png"),
                "msi_rgb": str(output_dir / "msi_rgb.png"),
                "overlay": str(output_dir / "overlay.png"),
                "legend": str(output_dir / "legend.png"),
            },
        }
        with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        prob_sum.flush()
        count.flush()
        del prob_sum
        del count

    if not args.keep_prob_memmap:
        for path in (prob_path, count_path):
            try:
                Path(path).unlink()
            except OSError:
                pass

    print("Done. Results saved to {}".format(output_dir.resolve()))


if __name__ == "__main__":
    main()
