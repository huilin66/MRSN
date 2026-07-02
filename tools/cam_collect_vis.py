#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate CAM visualizations inside collected sample folders.

This script is designed for folders produced by collect_top_miou_samples.py:

    ana/top20_xxx/
      rank_01_idx_431_miou_0.7523/
        rgb.png
        gt_color.png
        pred_color/

For each collected sample, it parses the validation index from the folder name,
loads each model, generates class-specific CAM overlays, and saves them to:

    rank_01_idx_431_miou_0.7523/cam_vis/cxup_4b_BW_PMRG_v2_lossV2_class_02_Street.png

By default it generates CAMs only for classes present in the sample GT.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PADDLECD_ROOT = REPO_ROOT / "PaddleCD"
if str(PADDLECD_ROOT) not in sys.path:
    sys.path.insert(0, str(PADDLECD_ROOT))


DEFAULT_MODELS = [
    "unet_BW",
    "deeplabv3p_BW",
    "ocrnet_BW",
    "segformer_BW",
    "highdan_BW",
    "cxup_1b_BW",
    "cxup_4b2h_BW",
    "cxup_4b_BW_PMRG_v2_lossV2",
]


@dataclass
class SampleInfo:
    path: Path
    rank: int | None
    index: int
    name: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate per-model, per-sample, per-class CAMs in collected folders."
    )
    parser.add_argument(
        "collect_dir",
        nargs="?",
        default="ana/top20_cxup_4b_BW_PMRG_v2_lossV2",
        type=Path,
        help="Collected sample directory. Default: ana/top20_cxup_4b_BW_PMRG_v2_lossV2",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model names. Default: comparison models used in the analysis.",
    )
    parser.add_argument(
        "--config_template",
        default="PaddleCD/c2seg_config/{model}.yml",
        help="Config path template. Default: PaddleCD/c2seg_config/{model}.yml",
    )
    parser.add_argument(
        "--model_template",
        default="output/{model}/best_model/model.pdparams",
        help="Model path template. Default: output/{model}/best_model/model.pdparams",
    )
    parser.add_argument(
        "--cam_layer",
        default="auto",
        help="Target layer name, or auto. Default: auto",
    )
    parser.add_argument(
        "--cam_target",
        choices=("gt_mask", "pred_mask", "full_image", "error_region"),
        default="gt_mask",
        help="Pixels used to aggregate the target class logit. Default: gt_mask",
    )
    parser.add_argument(
        "--class_mode",
        choices=("present_gt", "present_pred", "all", "select_file"),
        default="present_gt",
        help=(
            "Classes to visualize for each sample. present_gt is usually best for "
            "paper figures. Default: present_gt"
        ),
    )
    parser.add_argument(
        "--select_file",
        type=Path,
        help=(
            "Optional class list with lines like 'image_id: class_id/class_name; ...'. "
            "Used when --class_mode select_file."
        ),
    )
    parser.add_argument(
        "--select_id_type",
        choices=("auto", "rank", "index", "stem"),
        default="auto",
        help="How to interpret ids in --select_file. Default: auto.",
    )
    parser.add_argument(
        "--class_file",
        type=Path,
        default=Path("manuscript/class.txt"),
        help="Class-name file. Default: manuscript/class.txt",
    )
    parser.add_argument("--num_classes", type=int, default=14, help="Number of classes. Default: 14")
    parser.add_argument(
        "--rgb_bands",
        nargs=3,
        type=int,
        default=[0, 1, 2],
        help="Zero-based bands used for RGB overlay from image1. Default: 0 1 2",
    )
    parser.add_argument("--alpha", type=float, default=0.45, help="CAM opacity. Default: 0.45")
    parser.add_argument(
        "--save_cam_gray",
        action="store_true",
        help="Also save gray CAM maps directly under cam_vis.",
    )
    parser.add_argument(
        "--output_layout",
        choices=("flat", "class_dir"),
        default="flat",
        help=(
            "CAM file layout. flat saves cam_vis/<model>_class_XX_name.png; "
            "class_dir saves cam_vis/class_XX_name/<model>.png. Default: flat"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing CAM files.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Optional limit for quick tests. 0 means no limit.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Device for inference. Default: auto.",
    )
    parser.add_argument(
        "--data_format",
        choices=("NCHW", "NHWC"),
        default="NCHW",
        help="Input data format. Same constraint as PaddleCD/val.py.",
    )
    return parser.parse_args()


def choose_device(requested):
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


def iter_named_sublayers(model):
    if hasattr(model, "named_sublayers"):
        return list(model.named_sublayers())
    return []


def find_layer(model, layer_name):
    layers = dict(iter_named_sublayers(model))
    if layer_name in layers:
        return layer_name, layers[layer_name]
    matches = [(name, layer) for name, layer in layers.items() if name.endswith(layer_name)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(name for name, _layer in matches[:20])
        raise ValueError("Layer name '{}' is ambiguous. Matches: {}".format(layer_name, names))
    raise ValueError("Layer '{}' was not found.".format(layer_name))


def auto_find_cam_layer(model):
    candidate_names = [
        "decode_head4b.bottleneck",
        "decode_head.bottleneck",
        "decode_head3b.bottleneck",
        "decode_head1b.bottleneck",
        "decode_head4b.ppm.bottleneck",
        "decode_head.ppm.bottleneck",
        # UNet: use the final decoder feature before the class prediction conv.
        "decode.up_sample_list.3.double_conv.1",
        "decode.up_sample_list.3.double_conv.0",
    ]
    layers = dict(iter_named_sublayers(model))
    for name in candidate_names:
        if name in layers:
            return name, layers[name]

    conv_candidates = []
    for name, layer in layers.items():
        if layer.__class__.__name__ == "Conv2D":
            conv_candidates.append((name, layer))
    if conv_candidates:
        return conv_candidates[-1]
    raise ValueError("Could not choose an automatic CAM layer.")


def first_4d_tensor(output):
    if hasattr(output, "shape") and len(output.shape) == 4:
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            tensor = first_4d_tensor(item)
            if tensor is not None:
                return tensor
    return None


class GradCAM:
    def __init__(self, layer):
        self.layer = layer
        self.activation = None
        self.gradient = None
        self.handle = None

    def _forward_hook(self, layer, inputs, output):
        activation = first_4d_tensor(output)
        if activation is None:
            raise RuntimeError("CAM layer output must contain a 4D tensor.")
        self.activation = activation

        def _grad_hook(grad):
            self.gradient = grad

        activation.register_hook(_grad_hook)

    def __enter__(self):
        if hasattr(self.layer, "register_forward_post_hook"):
            self.handle = self.layer.register_forward_post_hook(self._forward_hook)
        elif hasattr(self.layer, "register_forward_hook"):
            self.handle = self.layer.register_forward_hook(self._forward_hook)
        else:
            raise RuntimeError("The selected layer does not support forward hooks.")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.handle is not None:
            self.handle.remove()

    def compute(self):
        if self.activation is None or self.gradient is None:
            raise RuntimeError("Missing CAM activation or gradient.")
        activation = self.activation.detach()
        gradient = self.gradient.detach()
        weights = paddle.mean(gradient, axis=[2, 3], keepdim=True)
        cam = paddle.sum(weights * activation, axis=1, keepdim=True)
        cam = paddle.nn.functional.relu(cam)
        return normalize_cam(cam.squeeze().numpy())


def get_model_logits(model, im1, im2, ori_shape=None):
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
    if ori_shape is not None and tuple(logit.shape[-2:]) != tuple(ori_shape):
        logit = paddle.nn.functional.interpolate(
            logit, size=ori_shape, mode="bilinear", align_corners=False
        )
    return logit


def build_target_mask(pred, label, class_id, target_mode, ignore_index):
    if target_mode == "full_image":
        return paddle.ones_like(pred).astype("float32")
    if target_mode == "pred_mask":
        mask = pred == class_id
    elif target_mode == "gt_mask":
        mask = label == class_id
    elif target_mode == "error_region":
        mask = (pred != label) & (label != ignore_index)
    else:
        raise ValueError("Unsupported CAM target mode: {}".format(target_mode))

    if int(mask.astype("int64").sum().numpy()) == 0:
        mask = paddle.ones_like(pred).astype("bool")
    return mask.astype("float32")


def cam_target_score(logits, pred, label, class_id, target_mode, ignore_index):
    mask = build_target_mask(pred, label, class_id, target_mode, ignore_index)
    class_logit = logits[:, class_id : class_id + 1, :, :]
    return paddle.sum(class_logit * mask) / (paddle.sum(mask) + 1e-6)


def normalize_cam(cam):
    cam = cam.astype("float32")
    cam = cam - cam.min()
    denom = cam.max()
    if denom > 0:
        cam = cam / denom
    return cam


def image_to_rgb(image, bands):
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[0] <= 16 and image.shape[-1] > 16:
        image = np.transpose(image, (1, 2, 0))

    max_band = image.shape[2] - 1
    selected = [min(max(band, 0), max_band) for band in bands]
    rgb = image[:, :, selected].astype("float32")

    channels = []
    for channel in range(3):
        band = rgb[:, :, channel]
        finite = np.isfinite(band)
        if not finite.any():
            channels.append(np.zeros_like(band, dtype="uint8"))
            continue
        lo, hi = np.percentile(band[finite], [2, 98])
        if hi <= lo:
            lo, hi = float(band[finite].min()), float(band[finite].max())
        if hi <= lo:
            scaled = np.zeros_like(band, dtype="uint8")
        else:
            scaled = np.clip((band - lo) / (hi - lo), 0, 1)
            scaled = (scaled * 255).astype("uint8")
        channels.append(scaled)
    return np.stack(channels, axis=-1)


def read_overlay_rgb(image_path, bands):
    suffix = Path(image_path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".bmp"}:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            image = skio.imread(image_path)
        elif image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = skio.imread(image_path)
    return image_to_rgb(image, bands)


def make_overlay(rgb, cam, alpha):
    cam_resized = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    cam_uint8 = (normalize_cam(cam_resized) * 255).astype("uint8")
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = rgb.astype("float32") * (1.0 - alpha) + heatmap.astype("float32") * alpha
    return np.clip(overlay, 0, 255).astype("uint8"), cam_uint8


def sanitize_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")


def parse_sample_dir(path):
    rank_match = re.search(r"rank_(\d+)", path.name)
    idx_match = re.search(r"_idx_([^_]+)", path.name)
    if not idx_match or not idx_match.group(1).isdigit():
        return None
    rank = int(rank_match.group(1)) if rank_match else None
    return SampleInfo(path=path, rank=rank, index=int(idx_match.group(1)), name=path.name)


def find_samples(collect_dir):
    samples = []
    for path in collect_dir.iterdir():
        if not path.is_dir() or not path.name.startswith("rank_"):
            continue
        sample = parse_sample_dir(path)
        if sample is not None:
            samples.append(sample)
    samples.sort(key=lambda item: (item.rank if item.rank is not None else 10**9, item.index))
    return samples


def parse_class_line(line):
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    match = re.match(r"^\s*\d+\s*[:,\s]\s*(.+)$", line)
    if match:
        return match.group(1).strip()
    return line


def load_class_names(path, num_classes):
    names = []
    if path and Path(path).is_file():
        with Path(path).open("r", encoding="utf-8-sig") as f:
            for line in f:
                name = parse_class_line(line)
                if name:
                    names.append(name)
    if len(names) < num_classes:
        names.extend("class_{}".format(i) for i in range(len(names), num_classes))
    return names[:num_classes]


def parse_class_specs(text):
    specs = []
    for part in re.split(r";+", text):
        part = part.strip()
        if not part:
            continue
        match = re.match(r"^\s*(\d+)\s*(?:[/：:,]\s*)?(.*)$", part)
        if not match:
            raise ValueError("Could not parse class spec '{}'.".format(part))
        specs.append(int(match.group(1)))
    return specs


def load_select_classes(select_file, samples, eval_dataset, id_type):
    by_rank = {
        str(sample.rank): sample.index
        for sample in samples
        if sample.rank is not None
    }
    by_rank.update({
        "{:02d}".format(sample.rank): sample.index
        for sample in samples
        if sample.rank is not None
    })
    by_index = {str(sample.index): sample.index for sample in samples}
    by_stem = {
        Path(eval_dataset.file_list[sample.index][0]).stem: sample.index
        for sample in samples
    }

    result = {}
    with Path(select_file).open("r", encoding="utf-8-sig") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            match = re.match(r"^\s*([^:：]+)\s*[:：]\s*(.+)$", line)
            if not match:
                raise ValueError("Invalid select_file line {}: {}".format(line_no, line))
            key = match.group(1).strip()
            classes = parse_class_specs(match.group(2))

            index = None
            if id_type in ("auto", "rank") and key in by_rank:
                index = by_rank[key]
            elif id_type in ("auto", "stem") and key in by_stem:
                index = by_stem[key]
            elif id_type in ("auto", "index") and key in by_index:
                index = by_index[key]
            if index is None:
                raise ValueError("Could not resolve select id '{}' at line {}.".format(key, line_no))
            result[index] = classes
    return result


def label_present_classes(label_np, num_classes, ignore_index, min_area=1):
    classes = []
    for class_id in range(num_classes):
        if class_id == ignore_index:
            continue
        if int((label_np == class_id).sum()) >= min_area:
            classes.append(class_id)
    return classes


def pred_present_classes(pred_np, num_classes, min_area=1):
    classes = []
    for class_id in range(num_classes):
        if int((pred_np == class_id).sum()) >= min_area:
            classes.append(class_id)
    return classes


def class_filename(model_name, class_id, class_names, prefix=""):
    class_name = class_names[class_id] if 0 <= class_id < len(class_names) else "class_{}".format(class_id)
    return "{}{}_class_{:02d}_{}.png".format(
        prefix,
        sanitize_name(model_name),
        class_id,
        sanitize_name(class_name),
    )


def class_folder_name(class_id, class_names):
    class_name = class_names[class_id] if 0 <= class_id < len(class_names) else "class_{}".format(class_id)
    return "class_{:02d}_{}".format(class_id, sanitize_name(class_name))


def cam_output_path(output_dir, model_name, class_id, class_names, args, prefix=""):
    if args.output_layout == "class_dir":
        class_dir = output_dir / class_folder_name(class_id, class_names)
        return class_dir / "{}{}.png".format(prefix, sanitize_name(model_name))
    return output_dir / class_filename(model_name, class_id, class_names, prefix=prefix)


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def reset_cam_vis_dirs(samples):
    for sample in samples:
        cam_dir = sample.path / "cam_vis"
        if cam_dir.exists():
            shutil.rmtree(cam_dir)
        cam_dir.mkdir(parents=True, exist_ok=True)


def resolve_model_paths(model_name, args):
    config_path = Path(args.config_template.format(model=model_name))
    model_path = Path(args.model_template.format(model=model_name))
    if not config_path.is_file():
        raise FileNotFoundError("Config not found for {}: {}".format(model_name, config_path))
    if not model_path.is_file():
        raise FileNotFoundError("Model params not found for {}: {}".format(model_name, model_path))
    return config_path, model_path


def load_model_context(model_name, args):
    config_path, model_path = resolve_model_paths(model_name, args)
    cfg = Config(str(config_path))
    apply_data_format(cfg, args.data_format)
    eval_dataset = cfg.val_dataset
    if eval_dataset is None:
        raise RuntimeError("Validation dataset is not specified in {}.".format(config_path))
    model = cfg.model
    utils.load_entire_model(model, str(model_path))
    model.eval()
    config_check(cfg, val_dataset=eval_dataset)
    if args.cam_layer == "auto":
        layer_name, cam_layer = auto_find_cam_layer(model)
    else:
        layer_name, cam_layer = find_layer(model, args.cam_layer)
    return cfg, eval_dataset, model, layer_name, cam_layer


def generate_for_sample_class(
    model,
    cam_layer,
    eval_dataset,
    sample,
    class_id,
    class_names,
    rgb,
    args,
    output_dir,
    model_name,
    layer_name,
):
    image1_path, image2_path, label_path = eval_dataset.file_list[sample.index]
    out_path = cam_output_path(output_dir, model_name, class_id, class_names, args)
    if out_path.is_file() and not args.overwrite:
        return {
            "sample": sample.name,
            "index": sample.index,
            "class_id": class_id,
            "class_name": class_names[class_id] if class_id < len(class_names) else "",
            "cam_layer": layer_name,
            "cam_target": args.cam_target,
            "cam_overlay_path": str(out_path),
            "skipped": True,
        }

    im1, im2, label = eval_dataset[sample.index]
    im1 = paddle.to_tensor(im1).unsqueeze(0)
    im2 = paddle.to_tensor(im2).unsqueeze(0)
    label = paddle.to_tensor(label).unsqueeze(0).astype("int64")

    model.clear_gradients()
    with GradCAM(cam_layer) as cam_runner:
        logits = get_model_logits(model, im1, im2, ori_shape=label.shape[-2:])
        pred = paddle.argmax(logits, axis=1, keepdim=True).astype("int32")
        score = cam_target_score(logits, pred, label, class_id, args.cam_target, eval_dataset.ignore_index)
        score.backward()
        cam = cam_runner.compute()

    overlay, cam_gray = make_overlay(rgb, cam, args.alpha)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(out_path)

    gray_path = ""
    if args.save_cam_gray:
        gray_path = cam_output_path(output_dir, model_name, class_id, class_names, args, prefix="gray_")
        gray_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(cam_gray).save(gray_path)

    model.clear_gradients()
    return {
        "sample": sample.name,
        "index": sample.index,
        "image1_path": image1_path,
        "image2_path": image2_path,
        "label_path": label_path,
        "class_id": class_id,
        "class_name": class_names[class_id] if class_id < len(class_names) else "",
        "cam_layer": layer_name,
        "cam_target": args.cam_target,
        "cam_overlay_path": str(out_path),
        "cam_gray_path": str(gray_path),
        "skipped": False,
    }


def main():
    args = parse_args()
    if not args.collect_dir.is_dir():
        raise FileNotFoundError("Collect directory not found: {}".format(args.collect_dir))

    global np
    global cv2
    global paddle
    global skio
    global Image
    global Config
    global config_check
    global get_sys_env
    global utils

    import cv2
    import numpy as np
    import paddle
    from PIL import Image
    from skimage import io as skio
    from paddleseg.cvlibs import Config
    from paddleseg.utils import config_check, get_sys_env, utils

    paddle.set_device(choose_device(args.device))

    samples = find_samples(args.collect_dir)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    if not samples:
        raise ValueError("No rank_* sample folders found in {}".format(args.collect_dir))

    reset_cam_vis_dirs(samples)

    class_names = load_class_names(args.class_file, args.num_classes)
    all_rows = []

    select_classes = None
    if args.class_mode == "select_file":
        if not args.select_file:
            raise ValueError("--select_file is required when --class_mode select_file.")

    print("Samples: {}".format(len(samples)))
    print("Models : {}".format(", ".join(args.models)))

    base_eval_dataset = None
    if args.class_mode == "select_file":
        # Built after the first model config is loaded, because it needs file_list.
        pass

    for model_idx, model_name in enumerate(args.models, start=1):
        print("\n[{}/{}] Loading model {}".format(model_idx, len(args.models), model_name))
        _cfg, eval_dataset, model, layer_name, cam_layer = load_model_context(model_name, args)
        print("Using CAM layer: {}".format(layer_name))

        if base_eval_dataset is None:
            base_eval_dataset = eval_dataset
            if args.class_mode == "select_file":
                select_classes = load_select_classes(
                    args.select_file, samples, base_eval_dataset, args.select_id_type
                )

        for sample_idx, sample in enumerate(samples, start=1):
            if sample.index >= len(eval_dataset):
                raise IndexError(
                    "Sample index {} from {} is outside validation dataset length {}.".format(
                        sample.index, sample.name, len(eval_dataset)
                    )
                )

            image1_path = eval_dataset.file_list[sample.index][0]
            rgb = read_overlay_rgb(image1_path, args.rgb_bands)
            _im1, _im2, label = eval_dataset[sample.index]
            label_np = np.asarray(label).squeeze()

            if args.class_mode == "all":
                classes = list(range(args.num_classes))
            elif args.class_mode == "present_gt":
                classes = label_present_classes(label_np, args.num_classes, eval_dataset.ignore_index)
            elif args.class_mode == "select_file":
                classes = select_classes.get(sample.index, [])
            else:
                im1 = paddle.to_tensor(_im1).unsqueeze(0)
                im2 = paddle.to_tensor(_im2).unsqueeze(0)
                with paddle.no_grad():
                    logits = get_model_logits(model, im1, im2, ori_shape=label_np.shape[-2:])
                    pred_np = paddle.argmax(logits, axis=1).squeeze().numpy()
                classes = pred_present_classes(pred_np, args.num_classes)

            cam_vis_dir = sample.path / "cam_vis"
            sample_rows = []
            for class_id in classes:
                row = generate_for_sample_class(
                    model,
                    cam_layer,
                    eval_dataset,
                    sample,
                    class_id,
                    class_names,
                    rgb,
                    args,
                    cam_vis_dir,
                    model_name,
                    layer_name,
                )
                row["model"] = model_name
                sample_rows.append(row)
                all_rows.append(row)

            write_csv(sample.path / "cam_vis" / "cam_vis_meta.csv", sample_rows)
            print(
                "  Sample {}/{} {}: {} classes".format(
                    sample_idx, len(samples), sample.name, len(classes)
                )
            )

        del model
        if hasattr(paddle, "device") and hasattr(paddle.device, "cuda"):
            empty_cache = getattr(paddle.device.cuda, "empty_cache", None)
            if empty_cache is not None:
                empty_cache()

    write_csv(args.collect_dir / "cam_vis_meta.csv", all_rows)
    print("\nSaved CAMs under each sample's cam_vis directory.")
    print("Saved summary metadata to {}".format((args.collect_dir / "cam_vis_meta.csv").resolve()))


if __name__ == "__main__":
    main()
