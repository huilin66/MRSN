#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build paper-style full-scene prediction comparison figures.

The script reads full-scene inference outputs produced by
tools/infer_full_scene.py:

    ana/full_scene/BW_wuhan/<model_name>/prediction_color_preview.png

and combines them with the full-scene GT color map.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
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

DEFAULT_CLASS_NAMES = [
    "Background",
    "Surface water",
    "Street",
    "Urban Fabric",
    "Industrial, commercial, and transport",
    "Mine, dump, and construction sites",
    "Artificial, vegetated areas",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Forests",
    "Shrub",
    "Open spaces with no vegetation",
    "Inland wetlands",
]

DISPLAY_NAMES = {
    "unet": "UNet",
    "deeplabv3p": "DeepLabV3+",
    "ocrnet": "OCRNet",
    "segformer": "SegFormer",
    "highdan": "HighDAN",
    "mrsn": "MRSN",
    "cxup_1b": "1-branch",
    "cxup_2b": "2-branch",
    "cxup_3b": "3-branch",
    "cxup_4b": "4-branch",
    "final": "MRSFN",
    "gt": "GT",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("AB", "BW"), required=True)
    parser.add_argument("--scene", required=True, help="Scene name, e.g. wuhan/beijing/berlin/augsburg.")
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=Path("ana/full_scene"),
        help="Root containing <dataset>_<scene>/<model> inference outputs.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"\\158.132.186.40\isds\huilin\bdd\cp_data\C2Seg\src"),
        help="C2Seg src root containing tif_BW/tif_AB or legacy tif/. Used to find GT.",
    )
    parser.add_argument(
        "--layout",
        choices=("2x6", "3x4", "4x3", "auto", "auto-tight"),
        default="2x6",
        help=(
            "2x6: other models / branch-family row. "
            "3x4: semantic / branches / strong+GT rows. "
            "4x3: compact portrait layout. "
            "auto/auto-tight: choose a tight layout from 2x6/3x4/4x3."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG. Default: <prediction-root>/<dataset>_<scene>/comparison_<layout>.png",
    )
    parser.add_argument("--tile-width", type=int, default=360)
    parser.add_argument("--tile-height", type=int, default=500)
    parser.add_argument(
        "--tight",
        action="store_true",
        help="Use scene aspect ratio to remove inner white padding for an explicit layout.",
    )
    parser.add_argument(
        "--auto-tile-height",
        action="store_true",
        help="Set tile height from the GT scene aspect ratio to reduce inner white padding.",
    )
    parser.add_argument(
        "--auto-tight",
        action="store_true",
        help="Alias for --tight, intended for compact paper figures.",
    )
    parser.add_argument(
        "--min-tile-height",
        type=int,
        default=140,
        help="Minimum tile height used with --auto-tile-height. Default: 140.",
    )
    parser.add_argument(
        "--tight-target-aspect",
        type=float,
        default=1.5,
        help="Target canvas aspect ratio used by --layout auto-tight. Default: 1.5.",
    )
    parser.add_argument("--gap", type=int, default=10)
    parser.add_argument("--margin", type=int, default=18)
    parser.add_argument("--header-height", type=int, default=44)
    parser.add_argument("--title-font-size", type=int, default=26)
    parser.add_argument("--legend-font-size", type=int, default=18)
    parser.add_argument("--num-classes", type=int, default=14)
    parser.add_argument("--class-file", type=Path, help="Optional class-name txt file.")
    parser.add_argument("--no-legend", action="store_true")
    parser.add_argument(
        "--mask-background",
        action="store_true",
        help="Mask invalid/background pixels using the full-scene label map before drawing tiles.",
    )
    parser.add_argument(
        "--background-class",
        type=int,
        default=0,
        help="Background class id used by --mask-background. Default: 0.",
    )
    parser.add_argument(
        "--masked-fill",
        default="white",
        help="Fill color outside the valid mask: white, gray, black, #RRGGBB, or R,G,B. Default: white.",
    )
    parser.add_argument(
        "--use-full-tif",
        action="store_true",
        help="Read prediction_color.tif instead of prediction_color_preview.png.",
    )
    parser.add_argument(
        "--final-model",
        help="Final model name. Default: cxup_4b_<dataset>_PMRG_v2_lossV2.",
    )
    parser.add_argument(
        "--mrsn-model",
        help="MRSN model name. Default: MRSN_<dataset>.",
    )
    return parser.parse_args()


def load_font(size: int, italic: bool = False, bold: bool = False):
    candidates = []
    if italic and bold:
        candidates.extend(["timesbi.ttf", "arialbi.ttf"])
    elif italic:
        candidates.extend(["timesi.ttf", "ariali.ttf"])
    elif bold:
        candidates.extend(["timesbd.ttf", "arialbd.ttf"])
    candidates.extend(["times.ttf", "arial.ttf", "DejaVuSerif.ttf", "DejaVuSans.ttf"])
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=2, align="center")
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_centered_text(draw, box, text, font, fill=(0, 0, 0)):
    x0, y0, x1, y1 = box
    w, h = text_size(draw, text, font)
    draw.multiline_text(
        (x0 + (x1 - x0 - w) / 2, y0 + (y1 - y0 - h) / 2),
        text,
        fill=fill,
        font=font,
        spacing=2,
        align="center",
    )


def canonical_scene(scene: str) -> str:
    return scene.lower()


def model_names(dataset: str, final_model: str | None, mrsn_model: str | None):
    suffix = dataset
    final = final_model or f"cxup_4b_{suffix}_PMRG_v2_lossV2"
    mrsn = mrsn_model or f"MRSN_{suffix}"
    return {
        "unet": f"unet_{suffix}",
        "deeplabv3p": f"deeplabv3p_{suffix}",
        "ocrnet": f"ocrnet_{suffix}",
        "segformer": f"segformer_{suffix}",
        "highdan": f"highdan_{suffix}",
        "mrsn": mrsn,
        "cxup_1b": f"cxup_1b_{suffix}",
        "cxup_2b": f"cxup_2b_{suffix}",
        "cxup_3b": f"cxup_3b_{suffix}",
        "cxup_4b": f"cxup_4b_{suffix}",
        "final": final,
    }


def layout_slots(layout: str):
    if layout == "2x6":
        return [
            ["unet", "deeplabv3p", "ocrnet", "segformer", "highdan", "mrsn"],
            ["cxup_1b", "cxup_2b", "cxup_3b", "cxup_4b", "final", "gt"],
        ]
    if layout == "3x4":
        return [
            ["unet", "deeplabv3p", "ocrnet", "segformer"],
            ["cxup_1b", "cxup_2b", "cxup_3b", "cxup_4b"],
            ["highdan", "mrsn", "final", "gt"],
        ]
    if layout == "4x3":
        return [
            ["unet", "deeplabv3p", "ocrnet"],
            ["segformer", "cxup_1b", "cxup_2b"],
            ["cxup_3b", "cxup_4b", "highdan"],
            ["mrsn", "final", "gt"],
        ]
    raise ValueError(f"Unsupported resolved layout: {layout}")


def resolve_layout(layout: str, scene_aspect: float, args) -> str:
    if layout not in {"auto", "auto-tight"}:
        return layout

    candidates = ("2x6", "3x4", "4x3")
    best_layout = candidates[0]
    best_score = float("inf")
    tile_h = max(args.min_tile_height, int(round(args.tile_width / scene_aspect)))
    for candidate in candidates:
        slots = layout_slots(candidate)
        rows = len(slots)
        cols = len(slots[0])
        grid_w = cols * args.tile_width + (cols - 1) * args.gap
        grid_h = rows * (args.header_height + tile_h) + (rows - 1) * args.gap
        width = args.margin * 2 + grid_w
        height = args.margin * 2 + grid_h + legend_height(width, args)
        score = abs(width / height - args.tight_target_aspect)
        if score < best_score:
            best_score = score
            best_layout = candidate
    return best_layout


def read_image(path: Path) -> Image.Image:
    if path.suffix.lower() in {".tif", ".tiff"}:
        arr = tifffile.memmap(path)
        arr = np.asarray(arr)
        return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    return Image.open(path).convert("RGB")


def parse_fill_color(value: str) -> tuple[int, int, int]:
    value = value.strip().lower()
    named = {
        "white": (255, 255, 255),
        "gray": (238, 238, 238),
        "grey": (238, 238, 238),
        "black": (0, 0, 0),
    }
    if value in named:
        return named[value]
    if value.startswith("#") and len(value) == 7:
        return tuple(int(value[i : i + 2], 16) for i in (1, 3, 5))
    parts = [part.strip() for part in value.split(",")]
    if len(parts) == 3:
        rgb = tuple(int(part) for part in parts)
        if all(0 <= channel <= 255 for channel in rgb):
            return rgb
    raise ValueError(f"Unsupported --masked-fill color: {value}")


def read_label_array(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        arr = np.asarray(tifffile.memmap(path))
    else:
        arr = np.asarray(Image.open(path))
    if arr.ndim == 3 and arr.shape[0] in {3, 4}:
        arr = np.moveaxis(arr, 0, -1)
    return arr


def valid_mask_from_label(path: Path, background_class: int) -> np.ndarray:
    label = read_label_array(path)
    if label.ndim == 2:
        return label != background_class
    if label.ndim == 3 and label.shape[-1] >= 3:
        background_color = np.asarray(BRIGHT_COLORS[background_class], dtype=label.dtype)
        return np.any(label[..., :3] != background_color, axis=-1)
    raise ValueError(f"Unsupported label shape for background mask: {path} {label.shape}")


def apply_valid_mask(image: Image.Image, mask: np.ndarray, fill: tuple[int, int, int]) -> Image.Image:
    if mask.shape != (image.height, image.width):
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        mask_image = mask_image.resize(image.size, Image.Resampling.NEAREST)
        mask = np.asarray(mask_image) > 0
    arr = np.asarray(image.convert("RGB")).copy()
    arr[~mask] = fill
    return Image.fromarray(arr, mode="RGB")


def fit_image(image: Image.Image, tile_size: tuple[int, int]) -> Image.Image:
    tile_w, tile_h = tile_size
    fitted = Image.new("RGB", (tile_w, tile_h), "white")
    src = image.copy()
    src.thumbnail((tile_w, tile_h), Image.Resampling.NEAREST)
    x = (tile_w - src.width) // 2
    y = (tile_h - src.height) // 2
    fitted.paste(src, (x, y))
    return fitted


def image_size(path: Path) -> tuple[int, int]:
    if path.suffix.lower() in {".tif", ".tiff"}:
        shape = tifffile.memmap(path).shape
        if len(shape) == 3 and shape[0] in {3, 4}:
            return int(shape[2]), int(shape[1])
        return int(shape[1]), int(shape[0])
    with Image.open(path) as image:
        return image.size


def auto_tile_height(gt_path: Path, tile_width: int, min_height: int) -> int:
    width, height = image_size(gt_path)
    if width <= 0:
        raise ValueError(f"Invalid GT image width: {gt_path}")
    return max(min_height, int(round(tile_width * height / width)))


def find_prediction(scene_dir: Path, model_name: str, use_full_tif: bool) -> Path:
    candidates = []
    model_dir = scene_dir / model_name
    if use_full_tif:
        candidates.append(model_dir / "prediction_color.tif")
    candidates.extend(
        [
            model_dir / "prediction_color_preview.png",
            model_dir / "prediction_color.png",
            model_dir / "prediction_color.tif",
        ]
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing prediction for {model_name}. Tried: {candidates}")


def find_gt(args) -> Path:
    scene = canonical_scene(args.scene)
    roots = [
        args.data_root / f"tif_{args.dataset}",
        args.data_root / ("tif_AB" if args.dataset == "AB" else "tif_BW"),
        args.data_root / "tif",
        args.data_root / f"C2Seg_{args.dataset}" / "tif",
        args.data_root / f"C2Seg_{args.dataset}" / f"tif_{args.dataset}",
    ]
    names = [
        f"{scene}_label_color.tif",
        f"{scene}_label_color.png",
        "thumbnails/" + f"{scene}_label_color_thumb.png",
        f"{scene}_label.tif",
    ]
    for root in roots:
        if not root.exists():
            continue
        for name in names:
            path = root / name
            if path.is_file():
                return path
        for path in root.glob(f"*{scene}*label*color*.tif"):
            return path
        for path in root.glob(f"*{scene}*label*color*.png"):
            return path
    raise FileNotFoundError(f"Could not find GT color image for {args.dataset} {args.scene}")


def find_label(args) -> Path:
    scene = canonical_scene(args.scene)
    roots = [
        args.data_root / f"tif_{args.dataset}",
        args.data_root / ("tif_AB" if args.dataset == "AB" else "tif_BW"),
        args.data_root / "tif",
        args.data_root / f"C2Seg_{args.dataset}" / "tif",
        args.data_root / f"C2Seg_{args.dataset}" / f"tif_{args.dataset}",
    ]
    names = [
        f"{scene}_label.tif",
        f"{scene}_label.png",
        f"{scene}_label_rawcode.tif",
        f"{scene}_label_rawcode.png",
        f"{scene}_label_color.tif",
        f"{scene}_label_color.png",
    ]
    for root in roots:
        if not root.exists():
            continue
        for name in names:
            path = root / name
            if path.is_file():
                return path
        for path in root.glob(f"*{scene}*label*.tif"):
            if "color" not in path.stem.lower():
                return path
        for path in root.glob(f"*{scene}*label*.png"):
            if "color" not in path.stem.lower():
                return path
    raise FileNotFoundError(f"Could not find label image for {args.dataset} {args.scene}")


def parse_class_line(line: str) -> str | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    match = re.match(r"^\s*\d+\s*[:,\s]\s*(.+)$", line)
    if match:
        return match.group(1).strip()
    return line


def load_class_names(path: Path | None, num_classes: int) -> list[str]:
    if path is None:
        default_path = Path("manuscript/class.txt")
        path = default_path if default_path.is_file() else None

    names = []
    if path and path.is_file():
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                name = parse_class_line(line)
                if name:
                    names.append(name)

    if not names:
        names = DEFAULT_CLASS_NAMES[:]

    if len(names) < num_classes:
        names.extend(f"class_{idx}" for idx in range(len(names), num_classes))
    return names[:num_classes]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> str:
    words = text.split()
    if not words:
        return text
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = current + " " + word
        if text_size(draw, candidate, font)[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return "\n".join(lines)


def legend_row_height(draw: ImageDraw.ImageDraw, image_w: int, args, margin: int) -> int:
    font = load_font(args.legend_font_size, italic=True)
    names = load_class_names(args.class_file, args.num_classes)
    rows = 2
    width = image_w - margin * 2
    cols = (len(names) + rows - 1) // rows
    cell_w = width // cols
    max_text_h = 0
    for name in names:
        wrapped = wrap_text(draw, name, font, max(20, cell_w - 12))
        max_text_h = max(max_text_h, text_size(draw, wrapped, font)[1])
    return max(46, max_text_h + 10)


def legend_height(image_w: int, args, margin: int | None = None) -> int:
    if args.no_legend:
        return 0
    margin = args.margin if margin is None else margin
    scratch = Image.new("RGB", (max(1, image_w), 1), "white")
    row_h = legend_row_height(ImageDraw.Draw(scratch), image_w, args, margin)
    return 18 + row_h * 2


def draw_legend(draw, image_w: int, y: int, args, margin: int) -> int:
    font = load_font(args.legend_font_size, italic=True)
    names = load_class_names(args.class_file, args.num_classes)
    rows = 2
    width = image_w - margin * 2
    x = margin
    cols = (len(names) + rows - 1) // rows
    cell_w = width // cols
    row_h = legend_row_height(draw, image_w, args, margin)
    for idx, name in enumerate(names):
        row = idx // cols
        col = idx % cols
        x0 = x + col * cell_w
        x1 = x + width if col == cols - 1 else x + (col + 1) * cell_w
        y0 = y + row * row_h
        y1 = y + (row + 1) * row_h
        color = BRIGHT_COLORS[idx] if idx < len(BRIGHT_COLORS) else (200, 200, 200)
        draw.rectangle([x0, y0, x1, y1], fill=color, outline=(60, 60, 60), width=1)
        luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        fill = (255, 255, 255) if luminance < 80 else (0, 0, 0)
        wrapped = wrap_text(draw, name, font, max(20, x1 - x0 - 12))
        draw_centered_text(draw, (x0 + 4, y0 + 2, x1 - 4, y1 - 2), wrapped, font, fill=fill)
    return math.ceil(len(names) / cols) * row_h


def build_figure(args):
    scene_tag = f"{args.dataset}_{canonical_scene(args.scene)}"
    scene_dir = args.prediction_root / scene_tag
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"Missing prediction scene directory: {scene_dir}")

    gt_path = find_gt(args)
    gt_w, gt_h = image_size(gt_path)
    if gt_h <= 0:
        raise ValueError(f"Invalid GT image height: {gt_path}")
    resolved_layout = resolve_layout(args.layout, gt_w / gt_h, args)

    names = model_names(args.dataset, args.final_model, args.mrsn_model)
    slots = layout_slots(resolved_layout)
    rows = len(slots)
    cols = len(slots[0])
    tile_w = args.tile_width
    use_auto_height = args.auto_tile_height or args.tight or args.auto_tight or args.layout in {"auto", "auto-tight"}
    tile_h = auto_tile_height(gt_path, tile_w, args.min_tile_height) if use_auto_height else args.tile_height
    tight_mode = args.tight or args.auto_tight or args.layout in {"auto", "auto-tight"}
    gap = min(args.gap, 6) if tight_mode else args.gap
    margin = min(args.margin, 10) if tight_mode else args.margin
    header_height = min(args.header_height, max(args.title_font_size + 6, 30)) if tight_mode else args.header_height
    grid_w = cols * tile_w + (cols - 1) * gap
    grid_h = rows * (header_height + tile_h) + (rows - 1) * gap
    width = margin * 2 + grid_w
    legend_h = legend_height(width, args, margin)
    height = margin * 2 + grid_h + legend_h

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(args.title_font_size, bold=True)
    valid_mask = None
    masked_fill = parse_fill_color(args.masked_fill)
    if args.mask_background:
        label_path = find_label(args)
        valid_mask = valid_mask_from_label(label_path, args.background_class)

    for r, row in enumerate(slots):
        for c, key in enumerate(row):
            x = margin + c * (tile_w + gap)
            y = margin + r * (header_height + tile_h + gap)
            title = DISPLAY_NAMES[key]
            draw_centered_text(draw, (x, y, x + tile_w, y + header_height), title, title_font)
            if key == "gt":
                path = gt_path
            else:
                path = find_prediction(scene_dir, names[key], args.use_full_tif)
            image = read_image(path)
            if valid_mask is not None:
                image = apply_valid_mask(image, valid_mask, masked_fill)
            tile = fit_image(image, (tile_w, tile_h))
            canvas.paste(tile, (x, y + header_height))

    if not args.no_legend:
        legend_y = margin + grid_h + 18
        draw_legend(draw, width, legend_y, args, margin)

    output_layout = f"{args.layout}_{resolved_layout}" if args.layout in {"auto", "auto-tight"} else args.layout
    suffix = "_tight" if tight_mode or args.auto_tile_height else ""
    output = args.output or scene_dir / f"comparison_{output_layout}{suffix}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(output)


def main():
    args = parse_args()
    build_figure(args)


if __name__ == "__main__":
    main()
