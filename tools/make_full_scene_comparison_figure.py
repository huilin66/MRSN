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
        choices=("2x6", "3x4"),
        default="2x6",
        help="2x6: other models / branch-family row. 3x4: semantic / MBB / strong+GT rows.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG. Default: <prediction-root>/<dataset>_<scene>/comparison_<layout>.png",
    )
    parser.add_argument("--tile-width", type=int, default=360)
    parser.add_argument("--tile-height", type=int, default=500)
    parser.add_argument("--gap", type=int, default=10)
    parser.add_argument("--margin", type=int, default=18)
    parser.add_argument("--header-height", type=int, default=44)
    parser.add_argument("--title-font-size", type=int, default=26)
    parser.add_argument("--legend-font-size", type=int, default=18)
    parser.add_argument("--num-classes", type=int, default=14)
    parser.add_argument("--class-file", type=Path, help="Optional class-name txt file.")
    parser.add_argument("--no-legend", action="store_true")
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
    return [
        ["unet", "deeplabv3p", "ocrnet", "segformer"],
        ["cxup_1b", "cxup_2b", "cxup_3b", "cxup_4b"],
        ["highdan", "mrsn", "final", "gt"],
    ]


def read_image(path: Path) -> Image.Image:
    if path.suffix.lower() in {".tif", ".tiff"}:
        arr = tifffile.memmap(path)
        arr = np.asarray(arr)
        return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    return Image.open(path).convert("RGB")


def fit_image(image: Image.Image, tile_size: tuple[int, int]) -> Image.Image:
    tile_w, tile_h = tile_size
    fitted = Image.new("RGB", (tile_w, tile_h), "white")
    src = image.copy()
    src.thumbnail((tile_w, tile_h), Image.Resampling.NEAREST)
    x = (tile_w - src.width) // 2
    y = (tile_h - src.height) // 2
    fitted.paste(src, (x, y))
    return fitted


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


def legend_height(image_w: int, args) -> int:
    if args.no_legend:
        return 0
    return 18 + max(92, args.legend_font_size * 4)


def draw_legend(draw, image_w: int, y: int, args) -> int:
    font = load_font(args.legend_font_size, italic=True)
    names = load_class_names(args.class_file, args.num_classes)
    rows = 2
    width = image_w - args.margin * 2
    x = args.margin
    cols = (len(names) + rows - 1) // rows
    cell_w = width // cols
    row_h = max(46, font.size * 2 + 12 if hasattr(font, "size") else 46)
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

    names = model_names(args.dataset, args.final_model, args.mrsn_model)
    slots = layout_slots(args.layout)
    rows = len(slots)
    cols = len(slots[0])
    tile_w, tile_h = args.tile_width, args.tile_height
    grid_w = cols * tile_w + (cols - 1) * args.gap
    grid_h = rows * (args.header_height + tile_h) + (rows - 1) * args.gap
    width = args.margin * 2 + grid_w
    legend_h = legend_height(width, args)
    height = args.margin * 2 + grid_h + legend_h

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(args.title_font_size, bold=True)

    gt_path = find_gt(args)
    for r, row in enumerate(slots):
        for c, key in enumerate(row):
            x = args.margin + c * (tile_w + args.gap)
            y = args.margin + r * (args.header_height + tile_h + args.gap)
            title = DISPLAY_NAMES[key]
            draw_centered_text(draw, (x, y, x + tile_w, y + args.header_height), title, title_font)
            if key == "gt":
                path = gt_path
            else:
                path = find_prediction(scene_dir, names[key], args.use_full_tif)
            tile = fit_image(read_image(path), (tile_w, tile_h))
            canvas.paste(tile, (x, y + args.header_height))

    if not args.no_legend:
        legend_y = args.margin + grid_h + 18
        draw_legend(draw, width, legend_y, args)

    output = args.output or scene_dir / f"comparison_{args.layout}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(output)


def main():
    args = parse_args()
    build_figure(args)


if __name__ == "__main__":
    main()
