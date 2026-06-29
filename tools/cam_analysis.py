"""Grad-CAM analysis for validation samples.

This script selects validation samples by per-image mIoU and saves one CAM
overlay on the RGB image for each selected sample.

Example:
    python tools/cam_analysis.py \
        --config PaddleCD/c2seg_config/MRSN.yml \
        --model_path output/iter_40000/model.pdparams \
        --output_dir tools/cam_results \
        --cam_class 1 \
        --topk 30
"""

import argparse
import csv
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PADDLECD_ROOT = REPO_ROOT / "PaddleCD"
if str(PADDLECD_ROOT) not in sys.path:
    sys.path.insert(0, str(PADDLECD_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM overlays for selected validation samples."
    )
    parser.add_argument(
        "--config", dest="cfg", required=True, help="Path to PaddleCD config file."
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to model params, for example output/iter_xxx/model.pdparams.",
    )
    parser.add_argument(
        "--output_dir",
        default="tools/cam_results",
        help="Directory for CAM overlays and metadata. Default: tools/cam_results",
    )
    parser.add_argument(
        "--cam_layer",
        default="auto",
        help="Target layer name. Use --list_layers to inspect names. Default: auto",
    )
    parser.add_argument(
        "--list_layers",
        action="store_true",
        help="Print model sublayer names and exit.",
    )
    parser.add_argument(
        "--cam_class",
        type=int,
        default=1,
        help="Class id used as CAM target. Default: 1",
    )
    parser.add_argument(
        "--cam_target",
        choices=("pred_mask", "gt_mask", "full_image", "error_region"),
        default="pred_mask",
        help="Pixels used to aggregate the target class logit. Default: pred_mask",
    )
    parser.add_argument(
        "--select_by",
        choices=("lowest_miou", "highest_miou", "all"),
        default="lowest_miou",
        help="Which validation samples to visualize. Default: lowest_miou",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=30,
        help="Number of selected samples for CAM when select_by is not all. Default: 30",
    )
    parser.add_argument(
        "--rgb_bands",
        nargs=3,
        type=int,
        default=[0, 1, 2],
        help="Zero-based bands used for RGB overlay from image1. Default: 0 1 2",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="CAM heatmap opacity in overlay. Default: 0.45",
    )
    parser.add_argument(
        "--save_cam_gray",
        action="store_true",
        help="Also save raw gray CAM maps.",
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
        raise ValueError(
            "Layer name '{}' is ambiguous. Matches: {}".format(layer_name, names)
        )
    raise ValueError("Layer '{}' was not found. Use --list_layers first.".format(layer_name))


def auto_find_cam_layer(model):
    candidate_names = [
        "decode_head4b.bottleneck",
        "decode_head.bottleneck",
        "decode_head3b.bottleneck",
        "decode_head1b.bottleneck",
        "decode_head4b.ppm.bottleneck",
        "decode_head.ppm.bottleneck",
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

    raise ValueError("Could not choose an automatic CAM layer. Use --list_layers.")


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


def predict_and_score(model, eval_dataset, sample_index):
    im1, im2, label = eval_dataset[sample_index]
    im1 = paddle.to_tensor(im1).unsqueeze(0)
    im2 = paddle.to_tensor(im2).unsqueeze(0)
    label = paddle.to_tensor(label).unsqueeze(0).astype("int64")
    logits = get_model_logits(model, im1, im2, ori_shape=label.shape[-2:])
    pred = paddle.argmax(logits, axis=1, keepdim=True).astype("int32")

    intersect_area, pred_area, label_area = metrics.calculate_area(
        pred, label, eval_dataset.num_classes, ignore_index=eval_dataset.ignore_index
    )
    class_iou, miou = metrics.mean_iou(intersect_area, pred_area, label_area)
    return {
        "index": sample_index,
        "miou": float(miou),
        "class_iou": class_iou.astype(float),
    }


def collect_sample_scores(model, eval_dataset):
    scores = []
    model.eval()
    with paddle.no_grad():
        for idx in range(len(eval_dataset)):
            scores.append(predict_and_score(model, eval_dataset, idx))
            print("Scored {}/{}".format(idx + 1, len(eval_dataset)))
    return scores


def select_samples(scores, select_by, topk):
    if select_by == "all":
        return scores
    reverse = select_by == "highest_miou"
    selected = sorted(scores, key=lambda item: item["miou"], reverse=reverse)
    return selected[:topk]


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
        cam = cam.squeeze().numpy()
        cam = normalize_cam(cam)
        return cam


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
    overlay = (rgb.astype("float32") * (1.0 - alpha) + heatmap.astype("float32") * alpha)
    return np.clip(overlay, 0, 255).astype("uint8"), cam_uint8


def output_name(sample_index, image_path):
    return "{:06d}_{}.png".format(sample_index, Path(image_path).stem)


def generate_cam_for_sample(model, eval_dataset, sample, cam_layer, args, output_dir):
    sample_index = sample["index"]
    image1_path, image2_path, label_path = eval_dataset.file_list[sample_index]
    im1, im2, label = eval_dataset[sample_index]
    im1 = paddle.to_tensor(im1).unsqueeze(0)
    im2 = paddle.to_tensor(im2).unsqueeze(0)
    label = paddle.to_tensor(label).unsqueeze(0).astype("int64")

    model.clear_gradients()
    cam_runner = GradCAM(cam_layer)
    with cam_runner:
        logits = get_model_logits(model, im1, im2, ori_shape=label.shape[-2:])
        pred = paddle.argmax(logits, axis=1, keepdim=True).astype("int32")
        score = cam_target_score(
            logits,
            pred,
            label,
            args.cam_class,
            args.cam_target,
            eval_dataset.ignore_index,
        )
        score.backward()
        cam = cam_runner.compute()

    rgb = read_overlay_rgb(image1_path, args.rgb_bands)
    overlay, cam_gray = make_overlay(rgb, cam, args.alpha)

    overlay_dir = output_dir / "cam_overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = overlay_dir / output_name(sample_index, image1_path)
    Image.fromarray(overlay).save(overlay_path)

    cam_gray_path = ""
    if args.save_cam_gray:
        cam_gray_dir = output_dir / "cam_gray"
        cam_gray_dir.mkdir(parents=True, exist_ok=True)
        cam_gray_path = cam_gray_dir / output_name(sample_index, image1_path)
        Image.fromarray(cam_gray).save(cam_gray_path)

    model.clear_gradients()
    return {
        "index": sample_index,
        "image1_path": image1_path,
        "image2_path": image2_path,
        "label_path": label_path,
        "miou": "{:.8f}".format(sample["miou"]),
        "cam_class": args.cam_class,
        "cam_target": args.cam_target,
        "cam_layer": args.resolved_cam_layer,
        "cam_overlay_path": str(overlay_path),
        "cam_gray_path": str(cam_gray_path),
    }


def write_meta(output_path, rows):
    fieldnames = [
        "index",
        "image1_path",
        "image2_path",
        "label_path",
        "miou",
        "cam_class",
        "cam_target",
        "cam_layer",
        "cam_overlay_path",
        "cam_gray_path",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()

    global np
    global cv2
    global paddle
    global skio
    global Image
    global Config
    global config_check
    global get_sys_env
    global metrics
    global utils

    import cv2
    import numpy as np
    import paddle
    from PIL import Image
    from skimage import io as skio
    from paddleseg.cvlibs import Config
    from paddleseg.utils import config_check, get_sys_env, metrics, utils

    paddle.set_device(choose_device(args.device))

    cfg = Config(args.cfg)
    apply_data_format(cfg, args.data_format)
    eval_dataset = cfg.val_dataset
    if eval_dataset is None:
        raise RuntimeError("The validation dataset is not specified in config.")
    if len(eval_dataset) == 0:
        raise ValueError("The validation dataset is empty.")

    model = cfg.model
    utils.load_entire_model(model, args.model_path)
    model.eval()
    config_check(cfg, val_dataset=eval_dataset)

    if args.list_layers:
        for name, layer in iter_named_sublayers(model):
            print("{}\t{}".format(name, layer.__class__.__name__))
        return

    if args.cam_layer == "auto":
        layer_name, cam_layer = auto_find_cam_layer(model)
    else:
        layer_name, cam_layer = find_layer(model, args.cam_layer)
    args.resolved_cam_layer = layer_name
    print("Using CAM layer: {}".format(layer_name))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = collect_sample_scores(model, eval_dataset)
    selected = select_samples(scores, args.select_by, args.topk)
    print("Selected {} samples for CAM.".format(len(selected)))

    rows = []
    for cam_index, sample in enumerate(selected):
        rows.append(generate_cam_for_sample(
            model, eval_dataset, sample, cam_layer, args, output_dir
        ))
        print("Generated CAM {}/{}".format(cam_index + 1, len(selected)))

    meta_path = output_dir / "cam_meta.csv"
    write_meta(meta_path, rows)
    print("Saved CAM overlays to {}".format(os.path.abspath(output_dir / "cam_overlay")))
    print("Saved CAM metadata to {}".format(os.path.abspath(meta_path)))


if __name__ == "__main__":
    main()
