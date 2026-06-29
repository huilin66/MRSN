"""Export per-image mIoU for the validation set.

Example:
    python tools/per_image_miou.py \
        --config PaddleCD/c2seg_config/MRSN.yml \
        --model_path output/iter_40000/model.pdparams \
        --batch_size 8 \
        --output tools/per_image_miou.csv \
        --output_dir tools/per_image_results \
        --pred_gray_dir tools/per_image_preds/gray \
        --pred_color_dir tools/per_image_preds/color
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
        description="Run validation and export mIoU for each image."
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
        "--output",
        default="tools/per_image_miou.csv",
        help="Output CSV path. Default: tools/per_image_miou.csv",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Directory for all outputs. If set, CSV is saved as "
            "<output_dir>/<output filename>, and predictions are saved under "
            "<output_dir>/<gray folder name> and <output_dir>/<color folder name>."
        ),
    )
    parser.add_argument(
        "--pred_gray_dir",
        default="tools/per_image_preds/gray",
        help="Directory for gray prediction masks. Default: tools/per_image_preds/gray",
    )
    parser.add_argument(
        "--pred_color_dir",
        default="tools/per_image_preds/color",
        help="Directory for pseudo-color prediction masks. Default: tools/per_image_preds/color",
    )
    parser.add_argument(
        "--no_save_pred",
        action="store_true",
        help="Only export metrics CSV, do not save prediction masks.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of DataLoader workers."
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Device for inference. Default: auto.",
    )

    parser.add_argument(
        "--aug_eval",
        action="store_true",
        help="Use multi-scale/flip augmentation during evaluation.",
    )
    parser.add_argument(
        "--swap",
        type=int,
        default=0,
        help="Swap mode used by augmented evaluation. Same as PaddleCD/val.py.",
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=[1.0],
        help="Scales for augmented evaluation.",
    )
    parser.add_argument(
        "--flip_horizontal",
        action="store_true",
        help="Use horizontal flip in augmented evaluation.",
    )
    parser.add_argument(
        "--flip_vertical",
        action="store_true",
        help="Use vertical flip in augmented evaluation.",
    )
    parser.add_argument(
        "--is_slide",
        action="store_true",
        help="Evaluate by sliding window.",
    )
    parser.add_argument(
        "--crop_size",
        nargs=2,
        type=int,
        default=None,
        help="Sliding window crop size: width height.",
    )
    parser.add_argument(
        "--stride",
        nargs=2,
        type=int,
        default=None,
        help="Sliding window stride: width height.",
    )
    parser.add_argument(
        "--data_format",
        choices=("NCHW", "NHWC"),
        default="NCHW",
        help="Input data format. Same constraint as PaddleCD/val.py.",
    )
    return parser.parse_args()


def resolve_output_paths(args):
    output_path = Path(args.output)
    gray_dir = Path(args.pred_gray_dir)
    color_dir = Path(args.pred_color_dir)

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_path = output_dir / output_path.name
        gray_dir = output_dir / gray_dir.name
        color_dir = output_dir / color_dir.name

    return output_path, gray_dir, color_dir


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


def build_test_config(cfg, args):
    test_config = cfg.test_config.copy()
    if args.aug_eval:
        test_config["aug_eval"] = True
        test_config["scales"] = args.scales
        test_config["swap"] = args.swap
    if args.flip_horizontal:
        test_config["flip_horizontal"] = True
    if args.flip_vertical:
        test_config["flip_vertical"] = True
    if args.is_slide:
        test_config["is_slide"] = True
        test_config["crop_size"] = args.crop_size
        test_config["stride"] = args.stride
    return test_config


def calculate_single_image_metrics(pred, label, num_classes, ignore_index):
    intersect_area, pred_area, label_area = metrics.calculate_area(
        pred, label, num_classes, ignore_index=ignore_index
    )
    class_iou, miou = metrics.mean_iou(intersect_area, pred_area, label_area)
    _class_acc, acc = metrics.accuracy(intersect_area, pred_area)
    kappa = metrics.kappa(intersect_area, pred_area, label_area)
    return {
        "miou": float(miou),
        "acc": float(acc),
        "kappa": float(kappa),
        "class_iou": class_iou.astype(float),
        "intersect": intersect_area.numpy().astype(np.int64),
        "pred_area": pred_area.numpy().astype(np.int64),
        "label_area": label_area.numpy().astype(np.int64),
    }


def infer_batch(model, eval_dataset, im1, im2, label, test_config):
    label = label.astype("int64")
    ori_shape = label.shape[-2:]
    transforms = eval_dataset.transforms.transforms

    if test_config.get("aug_eval", False):
        return infer.aug_inference(
            model,
            im1,
            im2,
            swap=test_config.get("swap", False),
            ori_shape=ori_shape,
            transforms=transforms,
            scales=test_config.get("scales", 1.0),
            flip_horizontal=test_config.get("flip_horizontal", True),
            flip_vertical=test_config.get("flip_vertical", False),
            is_slide=test_config.get("is_slide", False),
            stride=test_config.get("stride", None),
            crop_size=test_config.get("crop_size", None),
        )

    return infer.inference(
        model,
        im1,
        im2,
        ori_shape=ori_shape,
        transforms=transforms,
        stride=test_config.get("stride", None),
        crop_size=test_config.get("crop_size", None),
    )


def build_prediction_name(sample_index, image1_path, label_path):
    source_path = label_path or image1_path
    stem = Path(source_path).stem
    return "{:06d}_{}.png".format(sample_index, stem)


def save_prediction(pred, sample_index, image1_path, label_path, gray_dir, color_dir):
    pred_mask = pred.numpy().squeeze().astype("uint8")
    file_name = build_prediction_name(sample_index, image1_path, label_path)

    gray_path = Path(gray_dir) / file_name
    color_path = Path(color_dir) / file_name
    gray_path.parent.mkdir(parents=True, exist_ok=True)
    color_path.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(pred_mask).save(gray_path)
    get_pseudo_color_map(pred_mask).convert("RGB").save(color_path)

    return str(gray_path), str(color_path)


def write_rows(output_path, rows, num_classes):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "index",
        "image1_path",
        "image2_path",
        "label_path",
        "pred_gray_path",
        "pred_color_path",
        "miou",
        "acc",
        "kappa",
    ]
    fieldnames += ["class_{}_iou".format(i) for i in range(num_classes)]
    fieldnames += ["class_{}_intersect".format(i) for i in range(num_classes)]
    fieldnames += ["class_{}_pred_area".format(i) for i in range(num_classes)]
    fieldnames += ["class_{}_label_area".format(i) for i in range(num_classes)]

    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    output_path, pred_gray_dir, pred_color_dir = resolve_output_paths(args)

    global np
    global paddle
    global infer
    global Config
    global config_check
    global get_sys_env
    global metrics
    global utils
    global Image
    global get_pseudo_color_map

    import numpy as np
    import paddle
    from PIL import Image
    from paddleseg.core import infer
    from paddleseg.cvlibs import Config
    from paddleseg.utils import config_check, get_sys_env, metrics, utils
    from paddleseg.utils.visualize import get_pseudo_color_map

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
    test_config = build_test_config(cfg, args)

    batch_sampler = paddle.io.BatchSampler(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    loader = paddle.io.DataLoader(
        eval_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        return_list=True,
    )

    rows = []
    sample_offset = 0
    with paddle.no_grad():
        for batch_id, data in enumerate(loader):
            im1, im2, label = data
            pred = infer_batch(model, eval_dataset, im1, im2, label, test_config)
            label = label.astype("int64")

            batch_size = label.shape[0]
            for i in range(batch_size):
                sample_index = sample_offset + i
                image1_path, image2_path, label_path = eval_dataset.file_list[
                    sample_index
                ]
                item_metrics = calculate_single_image_metrics(
                    pred[i : i + 1],
                    label[i : i + 1],
                    eval_dataset.num_classes,
                    eval_dataset.ignore_index,
                )
                pred_gray_path = ""
                pred_color_path = ""
                if not args.no_save_pred:
                    pred_gray_path, pred_color_path = save_prediction(
                        pred[i],
                        sample_index,
                        image1_path,
                        label_path,
                        pred_gray_dir,
                        pred_color_dir,
                    )

                row = {
                    "index": sample_index,
                    "image1_path": image1_path,
                    "image2_path": image2_path,
                    "label_path": label_path,
                    "pred_gray_path": pred_gray_path,
                    "pred_color_path": pred_color_path,
                    "miou": "{:.8f}".format(item_metrics["miou"]),
                    "acc": "{:.8f}".format(item_metrics["acc"]),
                    "kappa": "{:.8f}".format(item_metrics["kappa"]),
                }
                for class_id, value in enumerate(item_metrics["class_iou"]):
                    row["class_{}_iou".format(class_id)] = "{:.8f}".format(value)
                for class_id, value in enumerate(item_metrics["intersect"]):
                    row["class_{}_intersect".format(class_id)] = int(value)
                for class_id, value in enumerate(item_metrics["pred_area"]):
                    row["class_{}_pred_area".format(class_id)] = int(value)
                for class_id, value in enumerate(item_metrics["label_area"]):
                    row["class_{}_label_area".format(class_id)] = int(value)
                rows.append(row)

            sample_offset += batch_size
            print(
                "Processed batch {}/{} ({}/{})".format(
                    batch_id + 1, len(loader), sample_offset, len(eval_dataset)
                )
            )

    write_rows(output_path, rows, eval_dataset.num_classes)
    print("Saved per-image mIoU to {}".format(os.path.abspath(output_path)))
    if not args.no_save_pred:
        print("Saved gray predictions to {}".format(os.path.abspath(pred_gray_dir)))
        print("Saved color predictions to {}".format(os.path.abspath(pred_color_dir)))


if __name__ == "__main__":
    main()
