#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactively annotate highlight boxes on collected prediction samples.

Default behavior:
  - reads <collect_dir>/select.txt if present
  - opens each selected sample's MRSFN prediction image
  - drag a rectangle, press s to save and move next
  - press n to skip, c to clear, q to quit

Output CSV columns:
    rank,idx,class_id,class_name,x1,y1,x2,y2,note,sample_dir,image_path

Coordinates are saved in the original single-image coordinate system.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image


DEFAULT_ANNOTATION_IMAGE = "pred_color/cxup_4b_BW_PMRG_v2_lossV2.png"


@dataclass
class SampleDir:
    path: Path
    rank: Optional[int]
    idx: Optional[str]


@dataclass
class AnnotItem:
    sample: SampleDir
    class_id: str = ""
    class_name: str = ""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactively annotate highlight boxes for comparison figures."
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
        help="Sample/class list. Default: <collect_dir>/select.txt if it exists.",
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        help="Sample ids/ranks, e.g. --ids 1 3 4 or --ids 1,3,4. Overrides --select-file.",
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_ANNOTATION_IMAGE,
        help="Relative image path inside each sample folder. Default: MRSFN pred_color.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path. Default: <collect_dir>/boxes.csv",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing output instead of overwriting.",
    )
    parser.add_argument(
        "--note",
        default="",
        help="Optional note written to each saved row.",
    )
    parser.add_argument(
        "--boundary-margin",
        type=int,
        default=4,
        help="Pixels used to judge whether a box touches image boundary. Default: 4",
    )
    return parser.parse_args()


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
    for sample_id in ids:
        key = str(sample_id).strip()
        sample = by_name.get(key) or by_rank.get(key.lstrip("0") or "0") or by_idx.get(key)
        if sample is not None:
            selected.append(sample)
        else:
            print("WARNING: missing sample id/rank: {}".format(sample_id))
    return selected


def parse_class_specs(text: str) -> list[tuple[str, str]]:
    specs = []
    for part in re.split(r";+", text):
        part = part.strip()
        if not part:
            continue
        match = re.match(r"^\s*(\d+)\s*(?:[/：:,]\s*)?(.*)$", part)
        if match:
            specs.append((match.group(1), match.group(2).strip()))
    return specs


def load_select_items(path: Path, samples: list[SampleDir]) -> list[AnnotItem]:
    by_rank = {str(item.rank): item for item in samples if item.rank is not None}
    by_idx = {str(item.idx): item for item in samples if item.idx is not None}
    by_name = {item.path.name: item for item in samples}
    items = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = re.match(r"^\s*([^:：]+)\s*[:：]\s*(.+)$", line)
            if match:
                sample_id = match.group(1).strip()
                class_specs = parse_class_specs(match.group(2))
            else:
                sample_id = re.split(r"\s+", line)[0]
                class_specs = [("", "")]

            sample = by_name.get(sample_id) or by_rank.get(sample_id.lstrip("0") or "0") or by_idx.get(sample_id)
            if sample is None:
                print("WARNING: missing sample id/rank: {}".format(sample_id))
                continue
            for class_id, class_name in class_specs:
                items.append(AnnotItem(sample=sample, class_id=class_id, class_name=class_name))
    return items


def build_items(args, samples: list[SampleDir]) -> list[AnnotItem]:
    if args.ids:
        return [AnnotItem(sample=sample) for sample in select_samples(samples, parse_id_tokens(args.ids))]
    select_file = args.select_file
    if select_file is None:
        default_select = args.collect_dir / "select.txt"
        select_file = default_select if default_select.is_file() else None
    if select_file is not None:
        return load_select_items(select_file, samples)
    return [AnnotItem(sample=sample) for sample in samples]


def boundary_info(x1, y1, x2, y2, width: int, height: int, margin: int):
    sides = []
    if x1 <= margin:
        sides.append("left")
    if y1 <= margin:
        sides.append("top")
    if x2 >= width - 1 - margin:
        sides.append("right")
    if y2 >= height - 1 - margin:
        sides.append("bottom")
    return bool(sides), "|".join(sides)


def row_from_item(item: AnnotItem, image_path: Path, rect, note: str, boundary_margin: int) -> dict[str, str]:
    x1, y1, x2, y2 = rect
    x_min, x_max = sorted((x1, x2))
    y_min, y_max = sorted((y1, y2))
    with Image.open(image_path) as img:
        width, height = img.size
    touch, sides = boundary_info(x_min, y_min, x_max, y_max, width, height, boundary_margin)
    return {
        "rank": "" if item.sample.rank is None else str(item.sample.rank),
        "idx": "" if item.sample.idx is None else str(item.sample.idx),
        "class_id": item.class_id,
        "class_name": item.class_name,
        "x1": str(int(round(x_min))),
        "y1": str(int(round(y_min))),
        "x2": str(int(round(x_max))),
        "y2": str(int(round(y_max))),
        "boundary_touch": "1" if touch else "0",
        "boundary_sides": sides,
        "note": note,
        "sample_dir": item.sample.path.name,
        "image_path": str(image_path),
    }


def write_rows(path: Path, rows: list[dict[str, str]], append: bool):
    fieldnames = [
        "rank",
        "idx",
        "class_id",
        "class_name",
        "x1",
        "y1",
        "x2",
        "y2",
        "boundary_touch",
        "boundary_sides",
        "note",
        "sample_dir",
        "image_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not append or not path.is_file()
    mode = "a" if append else "w"
    with path.open(mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def annotate_items(items: list[AnnotItem], args) -> list[dict[str, str]]:
    import matplotlib as mpl

    # Matplotlib binds "s"/"ctrl+s" to save-figure by default, which conflicts
    # with our "save annotation" shortcut.
    mpl.rcParams["keymap.save"] = []
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.widgets import RectangleSelector

    rows = []
    current_rect = {"value": None}

    for item_index, item in enumerate(items, start=1):
        image_path = item.sample.path / args.image
        if not image_path.is_file():
            print("WARNING: missing annotation image: {}".format(image_path))
            continue

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_title(
            "[{}/{}] {}  class {} {}\nDrag box, s=save, n=skip, c=clear, q=quit".format(
                item_index,
                len(items),
                item.sample.path.name,
                item.class_id,
                item.class_name,
            )
        )
        patch_holder = {"patch": None}

        def on_select(eclick, erelease):
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            if None in (x1, y1, x2, y2):
                return
            x1 = max(0, min(width - 1, x1))
            x2 = max(0, min(width - 1, x2))
            y1 = max(0, min(height - 1, y1))
            y2 = max(0, min(height - 1, y2))
            current_rect["value"] = (x1, y1, x2, y2)
            if patch_holder["patch"] is not None:
                patch_holder["patch"].remove()
            x_min, x_max = sorted((x1, x2))
            y_min, y_max = sorted((y1, y2))
            patch_holder["patch"] = Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                fill=False,
                edgecolor="red",
                linewidth=2,
            )
            ax.add_patch(patch_holder["patch"])
            fig.canvas.draw_idle()

        selector = RectangleSelector(
            ax,
            on_select,
            useblit=True,
            button=[1],
            minspanx=2,
            minspany=2,
            spancoords="pixels",
            interactive=True,
        )
        action = {"value": None}

        def on_key(event):
            if event.key == "s":
                if current_rect["value"] is None:
                    print("No box selected for {}; use n to skip.".format(item.sample.path.name))
                    return
                rows.append(row_from_item(
                    item,
                    image_path,
                    current_rect["value"],
                    args.note,
                    args.boundary_margin,
                ))
                action["value"] = "save"
                plt.close(fig)
            elif event.key == "n":
                action["value"] = "skip"
                plt.close(fig)
            elif event.key == "c":
                current_rect["value"] = None
                if patch_holder["patch"] is not None:
                    patch_holder["patch"].remove()
                    patch_holder["patch"] = None
                fig.canvas.draw_idle()
            elif event.key == "q":
                action["value"] = "quit"
                plt.close(fig)

        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()
        selector.set_active(False)
        if action["value"] == "quit":
            break
        current_rect["value"] = None
    return rows


def main():
    args = parse_args()
    if not args.collect_dir.is_dir():
        raise FileNotFoundError("Collect directory not found: {}".format(args.collect_dir))
    if args.output is None:
        args.output = args.collect_dir / "boxes.csv"

    samples = find_sample_dirs(args.collect_dir)
    items = build_items(args, samples)
    if not items:
        raise ValueError("No samples selected for annotation.")

    rows = annotate_items(items, args)
    if rows:
        write_rows(args.output, rows, args.append)
    print("Saved boxes : {}".format(len(rows)))
    print("Output CSV  : {}".format(args.output.resolve()))


if __name__ == "__main__":
    main()
