#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a paper-style segmentation comparison figure from collected samples.

The expected collected sample layout is produced by collect_top_miou_samples.py:

    sample_dir/
      rgb.png
      gt_color.png
      pred_color/<model_name>.png

Example:
    python tools/make_comparison_figure.py \
        ana/top20_cxup_4b_BW_PMRG_v2_lossV2 \
        --id-list ana/top20_cxup_4b_BW_PMRG_v2_lossV2/select.txt \
        --output ana/top20_cxup_4b_BW_PMRG_v2_lossV2/comparison.png
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont


DEFAULT_COLUMNS = [
    ("RGB", "rgb.png", "image"),
    ("GT", "gt_color.png", "mask"),
    ("UNet", "pred_color/unet_BW.png", "mask"),
    ("DeepLabV3+", "pred_color/deeplabv3p_BW.png", "mask"),
    ("OCRNet", "pred_color/ocrnet_BW.png", "mask"),
    ("SegFormer", "pred_color/segformer_BW.png", "mask"),
    ("HighDAN", "pred_color/highdan_BW.png", "mask"),
    ("UPerNet", "pred_color/cxup_1b_BW.png", "mask"),
    ("MRSN", "pred_color/cxup_4b2h_BW.png", "mask"),
    ("MBFM", "pred_color/cxup_4b_BW_PMRG_v2_lossV2.png", "mask"),
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


@dataclass
class SampleDir:
    path: Path
    rank: Optional[int]
    idx: Optional[str]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a Figure-10-style segmentation comparison grid."
    )
    parser.add_argument(
        "collect_dir",
        nargs="?",
        default="ana/top20_cxup_4b_BW_PMRG_v2_lossV2",
        type=Path,
        help="Collected sample directory. Default: ana/top20_cxup_4b_BW_PMRG_v2_lossV2",
    )
    parser.add_argument(
        "--id-list",
        type=Path,
        help="Text file with sample ids/ranks. Default: <collect_dir>/select.txt if it exists.",
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        help="Sample ids/ranks, e.g. --ids 1 3 4 or --ids 1,3,4. Overrides --id-list.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG path. Default: <collect_dir>/comparison_figure.png",
    )
    parser.add_argument("--tile-size", type=int, default=180, help="Tile size in pixels. Default: 180")
    parser.add_argument("--gap", type=int, default=8, help="Gap between tiles. Default: 8")
    parser.add_argument(
        "--header-height",
        type=int,
        default=48,
        help="Column header height in pixels. Default: 48",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=12,
        help="Outer margin in pixels. Default: 12",
    )
    parser.add_argument(
        "--title-font-size",
        type=int,
        default=28,
        help="Column title font size. Default: 28",
    )
    parser.add_argument(
        "--legend-font-size",
        type=int,
        default=20,
        help="Legend font size. Default: 20",
    )
    parser.add_argument(
        "--class-file",
        type=Path,
        help="Optional class-name text file. Default: manuscript/class.txt if it exists.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=14,
        help="Number of classes shown in legend. Default: 14",
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Do not draw the class legend at the bottom.",
    )
    parser.add_argument(
        "--boxes",
        type=Path,
        help=(
            "Optional boxes CSV. Coordinates are in original single-image pixels. "
            "Expected columns: rank or idx, x1, y1, x2, y2; optional class_id."
        ),
    )
    parser.add_argument(
        "--box-color",
        default="red",
        help="Highlight box color. Default: red.",
    )
    parser.add_argument(
        "--box-width",
        type=int,
        default=4,
        help="Highlight box width on output tiles. Default: 4.",
    )
    parser.add_argument(
        "--box-shadow-color",
        default="white",
        help="Thin contrast outline color for highlight boxes. Default: white.",
    )
    parser.add_argument(
        "--no-box-shadow",
        action="store_true",
        help="Draw only the main highlight box without the contrast outline.",
    )
    parser.add_argument(
        "--box-boundary-margin",
        type=int,
        default=4,
        help="Original-image pixels used to judge boundary boxes. Default: 4.",
    )
    parser.add_argument(
        "--no-box-boundary-inset",
        action="store_true",
        help="Do not move boundary-touching boxes inward; draw boxes exactly from CSV coordinates.",
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
    if not text:
        return 0, 0
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=2, align="center")
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def parse_sample_dir(path: Path) -> SampleDir:
    rank_match = re.search(r"rank_(\d+)", path.name)
    idx_match = re.search(r"_idx_([^_]+)", path.name)
    rank = int(rank_match.group(1)) if rank_match else None
    idx = idx_match.group(1) if idx_match else None
    return SampleDir(path=path, rank=rank, idx=idx)


def find_sample_dirs(collect_dir: Path) -> list[SampleDir]:
    dirs = [
        parse_sample_dir(path)
        for path in collect_dir.iterdir()
        if path.is_dir() and path.name.startswith("rank_")
    ]
    return sorted(dirs, key=lambda item: (item.rank if item.rank is not None else 10**9, natural_key(item.path)))


def parse_id_tokens(values: list[str]) -> list[str]:
    tokens = []
    for value in values:
        for part in re.split(r"[,;\s]+", value.strip()):
            if part:
                tokens.append(part)
    return tokens


def load_id_list(path: Path) -> list[str]:
    ids = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = re.match(r"\s*(\d+)", line)
            if match:
                ids.append(match.group(1))
            else:
                ids.append(line.split()[0])
    return ids


def select_samples(samples: list[SampleDir], ids: list[str]) -> list[SampleDir]:
    if not ids:
        return samples

    by_rank = {str(item.rank): item for item in samples if item.rank is not None}
    by_idx = {str(item.idx): item for item in samples if item.idx is not None}
    by_name = {item.path.name: item for item in samples}
    selected = []
    missing = []
    for sample_id in ids:
        key = str(sample_id).strip()
        sample = by_name.get(key) or by_rank.get(key.lstrip("0") or "0") or by_idx.get(key)
        if sample is None:
            rank_match = re.match(r"rank_(\d+)", key)
            if rank_match:
                sample = by_rank.get(str(int(rank_match.group(1))))
        if sample is None:
            missing.append(key)
        else:
            selected.append(sample)

    if missing:
        print("WARNING: missing sample ids/ranks: {}".format(", ".join(missing)), file=sys.stderr)
    return selected


def sample_key(sample: SampleDir) -> str:
    if sample.rank is not None:
        return str(sample.rank)
    if sample.idx is not None:
        return str(sample.idx)
    return sample.path.name


def parse_color(value: str) -> tuple[int, int, int]:
    named = {
        "red": (255, 0, 0),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "yellow": (255, 230, 0),
        "cyan": (0, 255, 255),
    }
    text = str(value).strip().lower()
    if text in named:
        return named[text]
    if text.startswith("#") and len(text) == 7:
        return tuple(int(text[i:i + 2], 16) for i in (1, 3, 5))
    parts = [part.strip() for part in text.split(",")]
    if len(parts) == 3:
        return tuple(max(0, min(255, int(part))) for part in parts)
    raise ValueError("Unsupported color '{}'. Use a name, #RRGGBB, or R,G,B.".format(value))


def box_row_key(row: dict[str, str]) -> Optional[str]:
    for key in ("rank", "sample_rank", "image_id", "id", "sample_id"):
        value = row.get(key)
        if value not in (None, ""):
            text = str(value).strip()
            if text.isdigit():
                return str(int(text))
            rank_match = re.search(r"rank_(\d+)", text)
            if rank_match:
                return str(int(rank_match.group(1)))
            return text
    for key in ("idx", "index", "sample_index"):
        value = row.get(key)
        if value not in (None, ""):
            return "idx:{}".format(str(value).strip())
    return None


def boundary_sides_for_box(
    box: tuple[float, float, float, float],
    source_size: tuple[int, int],
    margin: int,
) -> set[str]:
    x1, y1, x2, y2 = box
    source_w, source_h = source_size
    sides = set()
    if source_w <= 0 or source_h <= 0:
        return sides
    if x1 <= margin:
        sides.add("left")
    if y1 <= margin:
        sides.add("top")
    if x2 >= source_w - 1 - margin:
        sides.add("right")
    if y2 >= source_h - 1 - margin:
        sides.add("bottom")
    return sides


def load_boxes(path: Optional[Path]) -> dict[str, list[dict[str, object]]]:
    boxes: dict[str, list[dict[str, object]]] = {}
    if path is None:
        return boxes
    if not path.is_file():
        raise FileNotFoundError("Boxes CSV not found: {}".format(path))
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"x1", "y1", "x2", "y2"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("Boxes CSV missing columns: {}".format(", ".join(sorted(missing))))
        for row in reader:
            key = box_row_key(row)
            if key is None:
                continue
            x1, y1, x2, y2 = (float(row[name]) for name in ("x1", "y1", "x2", "y2"))
            x_min, x_max = sorted((x1, x2))
            y_min, y_max = sorted((y1, y2))
            sides = set()
            for side in str(row.get("boundary_sides", "")).split("|"):
                side = side.strip().lower()
                if side:
                    sides.add(side)
            boxes.setdefault(key, []).append({
                "coords": (x_min, y_min, x_max, y_max),
                "boundary_sides": sides,
            })
    return boxes


def boxes_for_sample(boxes: dict[str, list[tuple[float, float, float, float]]], sample: SampleDir):
    result = []
    if sample.rank is not None:
        result.extend(boxes.get(str(sample.rank), []))
    if sample.idx is not None:
        result.extend(boxes.get("idx:{}".format(sample.idx), []))
    result.extend(boxes.get(sample.path.name, []))
    return result


def draw_tile_boxes(
    draw: ImageDraw.ImageDraw,
    tile_box: tuple[int, int, int, int],
    boxes: list[dict[str, object]],
    source_size: tuple[int, int],
    color: tuple[int, int, int],
    shadow_color: tuple[int, int, int],
    width: int,
    boundary_margin: int,
    boundary_inset: bool,
):
    if not boxes:
        return
    x0, y0, x1, y1 = tile_box
    tile_w = x1 - x0
    tile_h = y1 - y0
    source_w, source_h = source_size
    if source_w <= 0 or source_h <= 0:
        return
    sx = tile_w / source_w
    sy = tile_h / source_h
    inset = max(width + 2, 1)
    for box_item in boxes:
        bx1, by1, bx2, by2 = box_item["coords"]
        rect = [
            int(round(x0 + bx1 * sx)),
            int(round(y0 + by1 * sy)),
            int(round(x0 + bx2 * sx)),
            int(round(y0 + by2 * sy)),
        ]
        if boundary_inset:
            sides = set(box_item.get("boundary_sides", set()))
            sides.update(boundary_sides_for_box((bx1, by1, bx2, by2), source_size, boundary_margin))
            if "left" in sides:
                rect[0] = max(rect[0], x0 + inset)
            if "top" in sides:
                rect[1] = max(rect[1], y0 + inset)
            if "right" in sides:
                rect[2] = min(rect[2], x1 - inset)
            if "bottom" in sides:
                rect[3] = min(rect[3], y1 - inset)
        if rect[2] <= rect[0] or rect[3] <= rect[1]:
            continue
        if shadow_color is not None:
            shadow_width = max(width + 2, 1)
            draw.rectangle(rect, outline=shadow_color, width=shadow_width)
        draw.rectangle(rect, outline=color, width=max(width, 1))


def image_size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return (0, 0)


def parse_class_line(line: str) -> Optional[str]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    match = re.match(r"^\s*\d+\s*[:,\s]\s*(.+)$", line)
    if match:
        return match.group(1).strip()
    return line


def load_class_names(path: Optional[Path], num_classes: int) -> list[str]:
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
        names.extend("class_{}".format(i) for i in range(len(names), num_classes))
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


def open_tile(path: Path, tile_size: int, kind: str) -> Image.Image:
    if not path.is_file():
        return missing_tile(tile_size, path.name)

    image = Image.open(path).convert("RGB")
    resample = Image.Resampling.LANCZOS if kind == "image" else Image.Resampling.NEAREST
    return image.resize((tile_size, tile_size), resample)


def missing_tile(tile_size: int, label: str) -> Image.Image:
    image = Image.new("RGB", (tile_size, tile_size), (238, 238, 238))
    draw = ImageDraw.Draw(image)
    font = load_font(max(12, tile_size // 12))
    text = "missing\n{}".format(label)
    w, h = text_size(draw, text, font)
    draw.text(((tile_size - w) / 2, (tile_size - h) / 2), text, fill=(160, 40, 40), font=font, align="center")
    return image


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font,
    fill=(0, 0, 0),
):
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


def draw_legend(
    image: Image.Image,
    x: int,
    y: int,
    width: int,
    class_names: list[str],
    font,
    rows: int = 2,
) -> int:
    draw = ImageDraw.Draw(image)
    rows = max(1, rows)
    cols = (len(class_names) + rows - 1) // rows
    cell_w = width // cols
    row_h = max(46, font.size * 2 + 12 if hasattr(font, "size") else 46)

    for idx, name in enumerate(class_names):
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
    return rows * row_h


def build_figure(samples: list[SampleDir], args) -> Image.Image:
    columns = DEFAULT_COLUMNS
    n_cols = len(columns)
    n_rows = len(samples)
    if n_rows == 0:
        raise ValueError("No samples selected.")

    tile = args.tile_size
    gap = args.gap
    margin = args.margin
    grid_w = n_cols * tile + (n_cols - 1) * gap
    grid_h = n_rows * tile + (n_rows - 1) * gap
    legend_gap = 18 if not args.no_legend else 0
    legend_h = 0 if args.no_legend else max(92, args.legend_font_size * 4)
    canvas_w = grid_w + margin * 2
    canvas_h = margin + args.header_height + grid_h + legend_gap + legend_h + margin

    image = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(args.title_font_size, italic=True, bold=True)
    legend_font = load_font(args.legend_font_size, italic=True)
    boxes_by_sample = load_boxes(args.boxes)
    box_color = parse_color(args.box_color)
    box_shadow_color = None if args.no_box_shadow else parse_color(args.box_shadow_color)

    grid_x = margin
    header_y = margin
    grid_y = margin + args.header_height

    for col_idx, (label, _rel_path, _kind) in enumerate(columns):
        x0 = grid_x + col_idx * (tile + gap)
        draw_centered_text(draw, (x0, header_y, x0 + tile, header_y + args.header_height), label, title_font)

    for row_idx, sample in enumerate(samples):
        y0 = grid_y + row_idx * (tile + gap)
        sample_boxes = boxes_for_sample(boxes_by_sample, sample)
        source_size = image_size(sample.path / "rgb.png")
        for col_idx, (_label, rel_path, kind) in enumerate(columns):
            x0 = grid_x + col_idx * (tile + gap)
            tile_img = open_tile(sample.path / rel_path, tile, kind)
            image.paste(tile_img, (x0, y0))
            if sample_boxes and rel_path.startswith("pred_color/"):
                draw_tile_boxes(
                    draw,
                    (x0, y0, x0 + tile, y0 + tile),
                    sample_boxes,
                    source_size,
                    box_color,
                    box_shadow_color,
                    args.box_width,
                    args.box_boundary_margin,
                    not args.no_box_boundary_inset,
                )

    if not args.no_legend:
        class_names = load_class_names(args.class_file, args.num_classes)
        legend_y = grid_y + grid_h + legend_gap
        draw_legend(image, grid_x, legend_y, grid_w, class_names, legend_font, rows=2)

    return image


def main():
    args = parse_args()
    collect_dir = args.collect_dir
    if not collect_dir.is_dir():
        raise FileNotFoundError("Collect directory not found: {}".format(collect_dir))

    if args.output is None:
        args.output = collect_dir / "comparison_figure.png"

    if args.ids:
        ids = parse_id_tokens(args.ids)
    else:
        id_list = args.id_list
        if id_list is None:
            default_id_list = collect_dir / "select.txt"
            id_list = default_id_list if default_id_list.is_file() else None
        ids = load_id_list(id_list) if id_list else []

    samples = select_samples(find_sample_dirs(collect_dir), ids)
    fig = build_figure(samples, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.save(args.output)

    print("Selected samples : {}".format(len(samples)))
    print("Output figure    : {}".format(args.output.resolve()))


if __name__ == "__main__":
    main()
