#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a paper-style CAM comparison figure from collected sample folders.

Expected layout:

    collect_dir/
      rank_01_idx_431_miou_0.7523/
        rgb.png
        gt_color.png
        cam_vis/
          unet_BW_class_02_Street.png
          deeplabv3p_BW_class_02_Street.png
          ...

By default this script reads <collect_dir>/select.txt, whose lines can be:

    1: 2/Street
    9: 2/Street;8/Permanent crops

Each sample/class pair becomes one row in the output figure.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont


DEFAULT_MODEL_COLUMNS = [
    ("UNet", "unet_BW"),
    ("DeepLabV3+", "deeplabv3p_BW"),
    ("OCRNet", "ocrnet_BW"),
    ("SegFormer", "segformer_BW"),
    ("HighDAN", "highdan_BW"),
    ("UPerNet", "cxup_1b_BW"),
    ("MRSN", "cxup_4b2h_BW"),
    ("MBFM", "cxup_4b_BW_PMRG_v2_lossV2"),
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


@dataclass
class RowSpec:
    sample: SampleDir
    class_id: int
    class_name: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a Figure-10-style CAM comparison grid."
    )
    parser.add_argument(
        "collect_dir",
        nargs="?",
        default="ana/top20_cxup_4b_BW_PMRG_v2_lossV2",
        type=Path,
        help="Collected sample directory. Default: ana/top20_cxup_4b_BW_PMRG_v2_lossV2",
    )
    parser.add_argument(
        "--select-file",
        type=Path,
        help="Sample/class file. Default: <collect_dir>/select.txt if it exists.",
    )
    parser.add_argument(
        "--select-cam-file",
        "--elect-cam-file",
        dest="select_cam_file",
        type=Path,
        help=(
            "Class-first CAM selection file. Lines accept 'class_id/class_name: rank ids' "
            "or just 'class_id/class_name' for all samples. Default: <collect_dir>/select_cam.txt if it exists."
        ),
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        help="Sample ids/ranks for fixed --cam-class mode, e.g. --ids 1 3 4 or --ids 1,3,4.",
    )
    parser.add_argument(
        "--cam-class",
        type=int,
        help="Fixed CAM class id for all selected samples. If omitted, class ids are read from select-file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG path. Default: <collect_dir>/cam_comparison_figure.png",
    )
    parser.add_argument("--tile-size", type=int, default=180, help="Tile size in pixels. Default: 180")
    parser.add_argument("--gap", type=int, default=8, help="Gap between tiles. Default: 8")
    parser.add_argument("--header-height", type=int, default=48, help="Column header height. Default: 48")
    parser.add_argument("--margin", type=int, default=12, help="Outer margin. Default: 12")
    parser.add_argument("--title-font-size", type=int, default=28, help="Column title font size. Default: 28")
    parser.add_argument("--label-font-size", type=int, default=16, help="Row label font size. Default: 16")
    parser.add_argument(
        "--label-width",
        type=int,
        default=110,
        help="Left row-label column width in pixels. Default: 110",
    )
    parser.add_argument("--legend-font-size", type=int, default=20, help="Legend font size. Default: 20")
    parser.add_argument(
        "--class-file",
        type=Path,
        help="Optional class-name text file. Default: manuscript/class.txt if it exists.",
    )
    parser.add_argument("--num-classes", type=int, default=14, help="Number of classes. Default: 14")
    parser.add_argument("--no-legend", action="store_true", help="Do not draw class legend.")
    parser.add_argument(
        "--no-row-label",
        action="store_true",
        help="Do not draw sample/class labels on the left side.",
    )
    parser.add_argument(
        "--no-rgb-gt",
        action="store_true",
        help="Only draw CAM columns; omit RGB and GT columns.",
    )
    return parser.parse_args()


def sanitize_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")


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
    samples = [
        parse_sample_dir(path)
        for path in collect_dir.iterdir()
        if path.is_dir() and path.name.startswith("rank_")
    ]
    return sorted(samples, key=lambda item: (item.rank if item.rank is not None else 10**9, natural_key(item.path)))


def parse_id_tokens(values: list[str]) -> list[str]:
    tokens = []
    for value in values:
        for part in re.split(r"[,;\s]+", value.strip()):
            if part:
                tokens.append(part)
    return tokens


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
            missing.append(key)
        else:
            selected.append(sample)
    if missing:
        print("WARNING: missing sample ids/ranks: {}".format(", ".join(missing)), file=sys.stderr)
    return selected


def parse_class_spec(text: str) -> list[tuple[int, str]]:
    specs = []
    for part in re.split(r";+", text):
        part = part.strip()
        if not part:
            continue
        match = re.match(r"^\s*(\d+)\s*(?:[/：:,]\s*)?(.*)$", part)
        if not match:
            raise ValueError("Could not parse class spec '{}'.".format(part))
        specs.append((int(match.group(1)), match.group(2).strip()))
    return specs


def load_select_file(path: Path) -> list[tuple[str, int, str]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            match = re.match(r"^\s*([^:：]+)\s*[:：]\s*(.+)$", line)
            if not match:
                raise ValueError("Invalid select file line {}: {}".format(line_no, line))
            sample_id = match.group(1).strip()
            for class_id, class_name in parse_class_spec(match.group(2)):
                rows.append((sample_id, class_id, class_name))
    return rows


def parse_select_cam_file(path: Path) -> list[tuple[int, str, list[str]]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                class_text, ids_text = line.split(":", 1)
            elif "：" in line:
                class_text, ids_text = line.split("：", 1)
            else:
                class_text, ids_text = line, "all"

            class_specs = parse_class_spec(class_text)
            if len(class_specs) != 1:
                raise ValueError(
                    "Invalid select_cam line {}: '{}'. Expected one class spec on the left.".format(
                        line_no, line
                    )
                )
            class_id, class_name = class_specs[0]
            ids_text = ids_text.strip()
            ids = [] if ids_text.lower() in {"", "all", "*"} else parse_id_tokens([ids_text])
            rows.append((class_id, class_name, ids))
    return rows


def sample_lookup(samples: list[SampleDir]):
    by_rank = {str(item.rank): item for item in samples if item.rank is not None}
    by_rank.update({"{:02d}".format(item.rank): item for item in samples if item.rank is not None})
    by_idx = {str(item.idx): item for item in samples if item.idx is not None}
    by_name = {item.path.name: item for item in samples}
    return by_rank, by_idx, by_name


def resolve_sample(samples: list[SampleDir], sample_id: str) -> Optional[SampleDir]:
    by_rank, by_idx, by_name = sample_lookup(samples)
    key = str(sample_id).strip()
    return by_name.get(key) or by_rank.get(key.lstrip("0") or "0") or by_idx.get(key)


def build_rows_from_select_cam_file(path: Path, samples: list[SampleDir], class_names: list[str]) -> list[RowSpec]:
    rows = []
    missing = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if ":" in line:
                left, right = line.split(":", 1)
            elif "：" in line:
                left, right = line.split("：", 1)
            else:
                left, right = line, "all"
            left = left.strip()
            right = right.strip()

            sample = resolve_sample(samples, left)
            if sample is not None:
                # Sample-first format: "rank_id: class_id/class_name; ..."
                for class_id, class_name in parse_class_spec(right):
                    if not class_name:
                        class_name = class_names[class_id] if class_id < len(class_names) else "class_{}".format(class_id)
                    rows.append(RowSpec(sample, class_id, class_name))
                continue

            # Class-first format: "class_id/class_name: rank ids" or "class_id/class_name".
            class_specs = parse_class_spec(left)
            if len(class_specs) != 1:
                raise ValueError(
                    "Invalid select_cam line {}: '{}'. Expected one class spec on the left.".format(
                        line_no, line
                    )
                )
            class_id, class_name = class_specs[0]
            if not class_name:
                class_name = class_names[class_id] if class_id < len(class_names) else "class_{}".format(class_id)
            ids = [] if right.lower() in {"", "all", "*"} else parse_id_tokens([right])
            selected = select_samples(samples, ids)
            if ids and not selected:
                missing.extend(ids)
            rows.extend(RowSpec(item, class_id, class_name) for item in selected)

    if missing:
        print("WARNING: missing sample ids/ranks: {}".format(", ".join(missing)), file=sys.stderr)
    return rows


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


def build_rows(samples: list[SampleDir], args, class_names: list[str]) -> list[RowSpec]:
    if args.cam_class is not None:
        ids = parse_id_tokens(args.ids or [])
        selected = select_samples(samples, ids)
        class_name = class_names[args.cam_class] if args.cam_class < len(class_names) else "class_{}".format(args.cam_class)
        return [RowSpec(sample, args.cam_class, class_name) for sample in selected]

    select_cam_file = args.select_cam_file
    if select_cam_file is None:
        default_select_cam = args.collect_dir / "select_cam.txt"
        select_cam_file = default_select_cam if default_select_cam.is_file() else None
    if select_cam_file is not None:
        return build_rows_from_select_cam_file(select_cam_file, samples, class_names)

    select_file = args.select_file
    if select_file is None:
        default_select = args.collect_dir / "select.txt"
        select_file = default_select if default_select.is_file() else None
    if select_file is None:
        raise ValueError("No --cam-class was given and no select.txt was found.")

    by_rank = {str(item.rank): item for item in samples if item.rank is not None}
    by_idx = {str(item.idx): item for item in samples if item.idx is not None}
    by_name = {item.path.name: item for item in samples}
    rows = []
    missing = []
    for sample_id, class_id, class_name in load_select_file(select_file):
        key = sample_id.strip()
        sample = by_name.get(key) or by_rank.get(key.lstrip("0") or "0") or by_idx.get(key)
        if sample is None:
            missing.append(key)
            continue
        if not class_name:
            class_name = class_names[class_id] if class_id < len(class_names) else "class_{}".format(class_id)
        rows.append(RowSpec(sample, class_id, class_name))
    if missing:
        print("WARNING: missing sample ids/ranks: {}".format(", ".join(missing)), file=sys.stderr)
    return rows


def cam_path_for(sample: SampleDir, model_name: str, class_id: int) -> Optional[Path]:
    cam_dir = sample.path / "cam_vis"
    class_dir_pattern = "class_{:02d}_*".format(class_id)
    for class_dir in sorted(cam_dir.glob(class_dir_pattern), key=natural_key):
        if not class_dir.is_dir():
            continue
        path = class_dir / "{}.png".format(sanitize_name(model_name))
        if path.is_file():
            return path

    pattern = "{}_class_{:02d}_*.png".format(sanitize_name(model_name), class_id)
    matches = sorted(cam_dir.glob(pattern), key=natural_key)
    if matches:
        return matches[0]
    fallback = cam_dir / "{}_class_{:02d}.png".format(sanitize_name(model_name), class_id)
    return fallback if fallback.is_file() else None


def open_tile(path: Optional[Path], tile_size: int, kind: str) -> Image.Image:
    if path is None or not path.is_file():
        return missing_tile(tile_size, path.name if path else "missing")
    image = Image.open(path).convert("RGB")
    resample = Image.Resampling.LANCZOS if kind in {"image", "cam"} else Image.Resampling.NEAREST
    return image.resize((tile_size, tile_size), resample)


def missing_tile(tile_size: int, label: str) -> Image.Image:
    image = Image.new("RGB", (tile_size, tile_size), (238, 238, 238))
    draw = ImageDraw.Draw(image)
    font = load_font(max(12, tile_size // 12))
    text = "missing\n{}".format(label)
    w, h = text_size(draw, text, font)
    draw.text(((tile_size - w) / 2, (tile_size - h) / 2), text, fill=(160, 40, 40), font=font, align="center")
    return image


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


def draw_legend(image: Image.Image, x: int, y: int, width: int, class_names: list[str], font, rows: int = 2) -> int:
    draw = ImageDraw.Draw(image)
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


def build_columns(args):
    columns = []
    if not args.no_rgb_gt:
        columns.extend([
            ("RGB", "rgb", None),
            ("GT", "gt", None),
        ])
    for title, model_name in DEFAULT_MODEL_COLUMNS:
        columns.append((title, "cam", model_name))
    return columns


def build_figure(rows: list[RowSpec], args, class_names: list[str]) -> Image.Image:
    columns = build_columns(args)
    if not rows:
        raise ValueError("No sample/class rows selected.")

    tile = args.tile_size
    gap = args.gap
    margin = args.margin
    label_w = 0 if args.no_row_label else max(0, args.label_width)
    grid_w = len(columns) * tile + (len(columns) - 1) * gap
    grid_h = len(rows) * tile + (len(rows) - 1) * gap
    legend_gap = 18 if not args.no_legend else 0
    legend_h = 0 if args.no_legend else max(92, args.legend_font_size * 4)
    canvas_w = margin * 2 + label_w + (gap if label_w else 0) + grid_w
    canvas_h = margin + args.header_height + grid_h + legend_gap + legend_h + margin

    image = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(args.title_font_size, italic=True, bold=True)
    label_font = load_font(args.label_font_size, italic=True)
    legend_font = load_font(args.legend_font_size, italic=True)

    grid_x = margin + label_w + (gap if label_w else 0)
    header_y = margin
    grid_y = margin + args.header_height

    for col_idx, (label, _kind, _model_name) in enumerate(columns):
        x0 = grid_x + col_idx * (tile + gap)
        draw_centered_text(draw, (x0, header_y, x0 + tile, header_y + args.header_height), label, title_font)

    for row_idx, row in enumerate(rows):
        y0 = grid_y + row_idx * (tile + gap)
        if label_w:
            label = wrap_text(draw, row.class_name, label_font, label_w - 12)
            draw_centered_text(draw, (margin, y0, margin + label_w, y0 + tile), label, label_font)

        for col_idx, (_label, kind, model_name) in enumerate(columns):
            x0 = grid_x + col_idx * (tile + gap)
            if kind == "rgb":
                path = row.sample.path / "rgb.png"
                tile_img = open_tile(path, tile, "image")
            elif kind == "gt":
                path = row.sample.path / "gt_color.png"
                tile_img = open_tile(path, tile, "mask")
            else:
                path = cam_path_for(row.sample, model_name, row.class_id)
                tile_img = open_tile(path, tile, "cam")
            image.paste(tile_img, (x0, y0))

    if not args.no_legend:
        legend_y = grid_y + grid_h + legend_gap
        draw_legend(image, grid_x, legend_y, grid_w, class_names, legend_font, rows=2)

    return image


def main():
    args = parse_args()
    if not args.collect_dir.is_dir():
        raise FileNotFoundError("Collect directory not found: {}".format(args.collect_dir))
    if args.output is None:
        args.output = args.collect_dir / "cam_comparison_figure.png"

    class_names = load_class_names(args.class_file, args.num_classes)
    samples = find_sample_dirs(args.collect_dir)
    rows = build_rows(samples, args, class_names)
    figure = build_figure(rows, args, class_names)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.save(args.output)

    print("Selected rows : {}".format(len(rows)))
    print("Output figure : {}".format(args.output.resolve()))


if __name__ == "__main__":
    main()
