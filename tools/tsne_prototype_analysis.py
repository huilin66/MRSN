"""GT/Pred region prototype t-SNE visualization.

Each point is one image-class prototype:
    prototype = mean(feature pixels inside one GT or prediction class region)

The default model path pattern follows aug.ipynb:
    config:     PaddleCD/c2seg_config/{model_name}.yml
    model_path: output/{model_name}/best_model/model.pdparams

Example:
    python tools/tsne_prototype_analysis.py \
        --model_names cxup_4b_BW cxup_4b_BW_PMRG_v2 \
        --output_dir results/prototype_tsne
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PADDLECD_ROOT = REPO_ROOT / "PaddleCD"
if str(PADDLECD_ROOT) not in sys.path:
    sys.path.insert(0, str(PADDLECD_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate GT and Pred prototype t-SNE plots for one or more models."
    )
    parser.add_argument(
        "--model_names",
        nargs="+",
        default=None,
        help=(
            "Model names using aug.ipynb path pattern: "
            "PaddleCD/c2seg_config/{name}.yml and output/{name}/best_model/model.pdparams."
        ),
    )
    parser.add_argument(
        "--models_json",
        default=None,
        help=(
            "Optional JSON list with fields name, config, model_path, and optional feature_layer. "
            "Overrides --model_names when set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="tools/prototype_tsne_results",
        help="Directory for figures, CSVs, and cached prototypes.",
    )
    parser.add_argument(
        "--redraw_from_cache",
        action="store_true",
        help=(
            "Only read prototypes_gt.npz/prototypes_pred.npz from output_dir and "
            "regenerate t-SNE figures/CSVs. This does not load models or extract features."
        ),
    )
    parser.add_argument(
        "--redraw_from_points",
        action="store_true",
        help=(
            "Only read prototype_gt_points.csv/prototype_pred_points.csv from output_dir and "
            "regenerate figures/CSVs. This keeps existing t-SNE coordinates and only updates "
            "plot styling/class names."
        ),
    )
    parser.add_argument(
        "--feature_layer",
        default="auto",
        help="Feature layer name for all models unless overridden by JSON. Default: auto",
    )
    parser.add_argument(
        "--list_layers",
        action="store_true",
        help="Print sublayer names for the first model and exit.",
    )
    parser.add_argument(
        "--min_pixels",
        type=int,
        default=20,
        help="Minimum pixels in a region to create one prototype. Default: 20",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Optional maximum number of validation images per model.",
    )
    parser.add_argument(
        "--max_points_per_class",
        type=int,
        default=None,
        help="Optional cap per source/model/class after prototype extraction.",
    )
    parser.add_argument(
        "--normalize",
        choices=("none", "l2", "standard"),
        default="l2",
        help="Feature normalization before t-SNE. Default: l2",
    )
    parser.add_argument(
        "--pca_dim",
        type=int,
        default=50,
        help="PCA dimension before t-SNE. Set 0 to disable PCA. Default: 50",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity. It will be reduced automatically for small sample counts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1919810,
        help="Random seed. Default: 1919810",
    )
    parser.add_argument(
        "--class_names",
        nargs="+",
        default=None,
        help="Optional class names, or one txt file with one class name per line.",
    )
    parser.add_argument(
        "--class_file",
        default=None,
        help=(
            "Class name txt file. Accepts lines like 'Water', '0: Water', "
            "'0,Water', or '0 Water'."
        ),
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


def load_model_specs(args):
    if args.models_json:
        with open(args.models_json, "r", encoding="utf-8") as f:
            specs = json.load(f)
        if not isinstance(specs, list):
            raise ValueError("--models_json must be a JSON list.")
        return specs

    if not args.model_names:
        raise ValueError("Please provide --model_names or --models_json.")

    specs = []
    for name in args.model_names:
        specs.append({
            "name": name,
            "config": "PaddleCD/c2seg_config/{}.yml".format(name),
            "model_path": "output/{}/best_model/model.pdparams".format(name),
            "feature_layer": args.feature_layer,
        })
    return specs


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


def auto_find_feature_layer(model):
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

    conv_candidates = [
        (name, layer)
        for name, layer in layers.items()
        if layer.__class__.__name__ == "Conv2D"
    ]
    if conv_candidates:
        return conv_candidates[-1]
    raise ValueError("Could not choose an automatic feature layer. Use --list_layers.")


def resolve_feature_layer(model, layer_name):
    if layer_name == "auto":
        return auto_find_feature_layer(model)
    return find_layer(model, layer_name)


def first_4d_tensor(output):
    if hasattr(output, "shape") and len(output.shape) == 4:
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            tensor = first_4d_tensor(item)
            if tensor is not None:
                return tensor
    return None


class FeatureHook:
    def __init__(self, layer):
        self.layer = layer
        self.feature = None
        self.handle = None

    def _hook(self, layer, inputs, output):
        feature = first_4d_tensor(output)
        if feature is None:
            raise RuntimeError("Feature layer output must contain a 4D tensor.")
        self.feature = feature

    def __enter__(self):
        if hasattr(self.layer, "register_forward_post_hook"):
            self.handle = self.layer.register_forward_post_hook(self._hook)
        elif hasattr(self.layer, "register_forward_hook"):
            self.handle = self.layer.register_forward_hook(self._hook)
        else:
            raise RuntimeError("The selected layer does not support forward hooks.")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.handle is not None:
            self.handle.remove()


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


def resize_mask(mask, target_hw):
    height, width = target_hw
    resized = cv2.resize(
        mask.astype("float32"),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype("int32")


def add_source_prototypes(records,
                          feature,
                          mask,
                          source,
                          model_name,
                          image_index,
                          image1_path,
                          image2_path,
                          label_path,
                          num_classes,
                          min_pixels,
                          ignore_index):
    for class_id in range(num_classes):
        region = mask == class_id
        pixel_count = int(region.sum())
        if pixel_count < min_pixels:
            continue

        proto = feature[:, region].mean(axis=1)
        records.append({
            "feature": proto.astype("float32"),
            "source": source,
            "model_name": model_name,
            "image_index": image_index,
            "image1_path": image1_path,
            "image2_path": image2_path,
            "label_path": label_path,
            "class_id": class_id,
            "pixel_count": pixel_count,
            "ignore_index": ignore_index,
        })


def extract_model_prototypes(spec, args):
    cfg = Config(spec["config"])
    apply_data_format(cfg, args.data_format)
    eval_dataset = cfg.val_dataset
    if eval_dataset is None:
        raise RuntimeError("No validation dataset in config: {}".format(spec["config"]))
    if len(eval_dataset) == 0:
        raise ValueError("The validation dataset is empty: {}".format(spec["config"]))

    model = cfg.model
    utils.load_entire_model(model, spec["model_path"])
    model.eval()
    config_check(cfg, val_dataset=eval_dataset)

    layer_name = spec.get("feature_layer") or args.feature_layer
    resolved_layer_name, feature_layer = resolve_feature_layer(model, layer_name)
    print("[{}] feature layer: {}".format(spec["name"], resolved_layer_name))

    records = []
    total = len(eval_dataset) if args.max_images is None else min(args.max_images, len(eval_dataset))
    with FeatureHook(feature_layer) as hook:
        with paddle.no_grad():
            for idx in range(total):
                image1_path, image2_path, label_path = eval_dataset.file_list[idx]
                im1, im2, label = eval_dataset[idx]
                im1 = paddle.to_tensor(im1).unsqueeze(0)
                im2 = paddle.to_tensor(im2).unsqueeze(0)
                label_np = np.asarray(label).squeeze().astype("int32")
                label_tensor = paddle.to_tensor(label).unsqueeze(0).astype("int64")

                logits = get_model_logits(
                    model, im1, im2, ori_shape=label_tensor.shape[-2:]
                )
                pred_np = paddle.argmax(logits, axis=1).numpy().squeeze().astype("int32")

                if hook.feature is None:
                    raise RuntimeError("Feature hook did not capture an activation.")
                feature = hook.feature.numpy().squeeze(0).astype("float32")
                feature_hw = feature.shape[-2:]
                gt_feature_mask = resize_mask(label_np, feature_hw)
                pred_feature_mask = resize_mask(pred_np, feature_hw)

                add_source_prototypes(
                    records,
                    feature,
                    gt_feature_mask,
                    "gt",
                    spec["name"],
                    idx,
                    image1_path,
                    image2_path,
                    label_path,
                    eval_dataset.num_classes,
                    args.min_pixels,
                    eval_dataset.ignore_index,
                )
                add_source_prototypes(
                    records,
                    feature,
                    pred_feature_mask,
                    "pred",
                    spec["name"],
                    idx,
                    image1_path,
                    image2_path,
                    label_path,
                    eval_dataset.num_classes,
                    args.min_pixels,
                    eval_dataset.ignore_index,
                )
                hook.feature = None
                print("[{}] extracted {}/{}".format(spec["name"], idx + 1, total))

    return records, {
        "name": spec["name"],
        "config": spec["config"],
        "model_path": spec["model_path"],
        "feature_layer": resolved_layer_name,
        "num_classes": eval_dataset.num_classes,
    }


def cap_points_per_class(records, max_points, seed):
    if max_points is None:
        return records

    rng = np.random.default_rng(seed)
    grouped = {}
    for index, record in enumerate(records):
        key = (record["source"], record["model_name"], record["class_id"])
        grouped.setdefault(key, []).append(index)

    keep_indices = []
    for indices in grouped.values():
        if len(indices) <= max_points:
            keep_indices.extend(indices)
        else:
            keep_indices.extend(rng.choice(indices, size=max_points, replace=False).tolist())
    keep_indices = sorted(keep_indices)
    return [records[index] for index in keep_indices]


def normalize_features(features, mode):
    if mode == "none":
        return features
    if mode == "l2":
        norm = np.linalg.norm(features, axis=1, keepdims=True)
        norm[norm == 0] = 1
        return features / norm
    if mode == "standard":
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True)
        std[std == 0] = 1
        return (features - mean) / std
    raise ValueError("Unsupported normalize mode: {}".format(mode))


def align_feature_dimensions(records, pca_dim, seed):
    dims = sorted({int(record["feature"].shape[0]) for record in records})
    if len(dims) == 1:
        return np.stack([record["feature"] for record in records], axis=0)

    grouped = {}
    for index, record in enumerate(records):
        grouped.setdefault(record["model_name"], []).append(index)

    common_dim = min([pca_dim] + dims)
    for indices in grouped.values():
        common_dim = min(common_dim, len(indices) - 1)
    common_dim = max(common_dim, 2)

    aligned = np.zeros((len(records), common_dim), dtype="float32")
    for model_name, indices in grouped.items():
        matrix = np.stack([records[index]["feature"] for index in indices], axis=0)
        local_dim = min(common_dim, matrix.shape[0] - 1, matrix.shape[1])
        if local_dim < 2:
            projected = matrix[:, :common_dim]
            if projected.shape[1] < common_dim:
                pad = common_dim - projected.shape[1]
                projected = np.pad(projected, ((0, 0), (0, pad)))
        else:
            projected = PCA(n_components=local_dim, random_state=seed).fit_transform(matrix)
            if local_dim < common_dim:
                projected = np.pad(projected, ((0, 0), (0, common_dim - local_dim)))
        aligned[indices, :] = projected.astype("float32")
        print(
            "Aligned feature dim for model {}: {} -> {}".format(
                model_name, matrix.shape[1], common_dim
            )
        )
    return aligned


def run_tsne(features, args):
    features = normalize_features(features.astype("float32"), args.normalize)
    if args.pca_dim and args.pca_dim > 0:
        pca_dim = min(args.pca_dim, features.shape[0] - 1, features.shape[1])
        if pca_dim >= 2:
            features = PCA(n_components=pca_dim, random_state=args.seed).fit_transform(features)

    if features.shape[0] < 3:
        raise ValueError("Need at least 3 prototype points for t-SNE.")

    perplexity = min(args.perplexity, max(1.0, (features.shape[0] - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=args.seed,
    )
    return tsne.fit_transform(features)


def read_text_auto(path):
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return Path(path).read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return Path(path).read_text(encoding="utf-8", errors="replace")


def load_class_names_file(path):
    names = []
    for raw_line in read_text_auto(path).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        match = re.match(r"^\s*\d+\s*(?:[:;,|\t]|\s)\s*(.+?)\s*$", line)
        if match:
            line = match.group(1).strip()
        names.append(line)
    return names


def resolve_class_names(class_names, class_file, num_classes):
    names = ["class_{}".format(i) for i in range(num_classes)]

    provided = []
    if class_file:
        class_path = Path(class_file)
        if not class_path.is_file():
            raise FileNotFoundError("Class file does not exist: {}".format(class_path))
        provided = load_class_names_file(class_path)
    elif class_names:
        if len(class_names) == 1 and Path(class_names[0]).exists():
            provided = load_class_names_file(Path(class_names[0]))
        else:
            provided = list(class_names)

    if provided:
        if len(provided) != num_classes:
            print(
                "WARNING: class names contain {} names, but num_classes is {}. "
                "Matching class ids will be replaced; unmatched ids keep class_i names.".format(
                    len(provided), num_classes
                )
            )
        for class_id, class_name in enumerate(provided[:num_classes]):
            names[class_id] = class_name

    return names


def apply_class_names_to_records(records, class_names):
    for record in records:
        class_id = int(record["class_id"])
        record["class_name"] = (
            class_names[class_id] if 0 <= class_id < len(class_names)
            else "class_{}".format(class_id)
        )


def marker_for_index(index):
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8"]
    return markers[index % len(markers)]


def plot_tsne(points, records, source, output_path, class_names):
    from matplotlib.lines import Line2D

    plt.figure(figsize=(10, 8), dpi=180)
    model_names = sorted({record["model_name"] for record in records})
    class_ids = sorted({int(record["class_id"]) for record in records})
    cmap = plt.get_cmap("tab20", max(len(class_ids), 1))
    class_to_color = {class_id: cmap(i % 20) for i, class_id in enumerate(class_ids)}
    model_to_marker = {
        model_name: marker_for_index(i) for i, model_name in enumerate(model_names)
    }

    for model_name in model_names:
        for class_id in class_ids:
            indices = [
                i
                for i, record in enumerate(records)
                if record["model_name"] == model_name and int(record["class_id"]) == class_id
            ]
            if not indices:
                continue
            label = None
            if len(model_names) == 1:
                label = class_names[class_id] if class_id < len(class_names) else "class_{}".format(class_id)
            elif class_id == class_ids[0]:
                label = model_name
            plt.scatter(
                points[indices, 0],
                points[indices, 1],
                s=28,
                c=[class_to_color[class_id]],
                marker=model_to_marker[model_name],
                edgecolors="none",
                alpha=0.82,
                label=label,
            )

    ax = plt.gca()
    if len(model_names) == 1:
        legend_title = "GT class" if source == "gt" else "Pred class"
        ax.legend(title=legend_title, fontsize=8, title_fontsize=9, loc="best", frameon=True)
    else:
        class_handles = []
        for class_id in class_ids:
            class_label = class_names[class_id] if class_id < len(class_names) else "class_{}".format(class_id)
            class_handles.append(Line2D(
                [0], [0], marker="o", linestyle="None", markersize=7,
                markerfacecolor=class_to_color[class_id], markeredgecolor="none",
                label=class_label))
        model_handles = []
        for model_name in model_names:
            model_handles.append(Line2D(
                [0], [0], marker=model_to_marker[model_name], linestyle="None",
                markersize=7, markerfacecolor="gray", markeredgecolor="none",
                label=model_name))
        class_legend = ax.legend(
            handles=class_handles,
            title="GT class" if source == "gt" else "Pred class",
            fontsize=7,
            title_fontsize=8,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=True)
        ax.add_artist(class_legend)
        ax.legend(
            handles=model_handles,
            title="Model",
            fontsize=7,
            title_fontsize=8,
            loc="lower left",
            bbox_to_anchor=(1.02, 0.0),
            frameon=True)
    plt.title("{} prototype t-SNE".format(source.upper()))
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)
    plt.tight_layout(rect=[0, 0, 0.78, 1] if len(model_names) > 1 else None)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def write_points_csv(output_path, points, records):
    fieldnames = [
        "x",
        "y",
        "source",
        "model_name",
        "image_index",
        "image1_path",
        "image2_path",
        "label_path",
        "class_id",
        "class_name",
        "pixel_count",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for point, record in zip(points, records):
            writer.writerow({
                "x": "{:.8f}".format(float(point[0])),
                "y": "{:.8f}".format(float(point[1])),
                "source": record["source"],
                "model_name": record["model_name"],
                "image_index": record["image_index"],
                "image1_path": record["image1_path"],
                "image2_path": record["image2_path"],
                "label_path": record["label_path"],
                "class_id": record["class_id"],
                "class_name": record.get("class_name", "class_{}".format(record["class_id"])),
                "pixel_count": record["pixel_count"],
            })


def save_prototype_cache(output_path, records):
    dims = {int(record["feature"].shape[0]) for record in records}
    if len(dims) == 1:
        features = np.stack([record["feature"] for record in records], axis=0)
    else:
        features = np.empty((len(records),), dtype=object)
        for index, record in enumerate(records):
            features[index] = record["feature"]
    meta = []
    for record in records:
        meta.append({
            key: value
            for key, value in record.items()
            if key != "feature"
        })
    np.savez_compressed(
        output_path,
        features=features,
        meta=np.array(json.dumps(meta, ensure_ascii=False), dtype=object),
    )


def load_prototype_cache(input_path):
    if not input_path.is_file():
        return []
    data = np.load(input_path, allow_pickle=True)
    features = data["features"]
    meta = json.loads(str(data["meta"].item()))
    records = []
    for index, item in enumerate(meta):
        record = dict(item)
        record["feature"] = np.asarray(features[index]).astype("float32")
        records.append(record)
    return records


def load_points_csv(input_path):
    if not input_path.is_file():
        return None, []

    points = []
    records = []
    with input_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            points.append([float(row["x"]), float(row["y"])])
            record = dict(row)
            record["class_id"] = int(record["class_id"])
            if "pixel_count" in record and record["pixel_count"] != "":
                record["pixel_count"] = int(float(record["pixel_count"]))
            records.append(record)

    if not points:
        return None, []
    return np.asarray(points, dtype="float32"), records


def redraw_from_points(output_dir, args):
    gt_points, gt_records = load_points_csv(output_dir / "prototype_gt_points.csv")
    pred_points, pred_records = load_points_csv(output_dir / "prototype_pred_points.csv")
    all_records = gt_records + pred_records

    if not all_records:
        raise RuntimeError(
            "No cached t-SNE points found in {}. Expected prototype_gt_points.csv or "
            "prototype_pred_points.csv.".format(output_dir)
        )

    num_classes = max(int(record["class_id"]) for record in all_records) + 1
    class_names = resolve_class_names(args.class_names, args.class_file, num_classes)
    apply_class_names_to_records(all_records, class_names)

    results = []
    if gt_records:
        fig_path = output_dir / "tsne_gt_by_class.png"
        csv_path = output_dir / "prototype_gt_points.csv"
        plot_tsne(gt_points, gt_records, "gt", fig_path, class_names)
        write_points_csv(csv_path, gt_points, gt_records)
        results.append({
            "source": "gt",
            "count": len(gt_records),
            "figure": str(fig_path),
            "csv": str(csv_path),
        })
        print("Redrew gt t-SNE figure to {}".format(os.path.abspath(fig_path)))
    if pred_records:
        fig_path = output_dir / "tsne_pred_by_class.png"
        csv_path = output_dir / "prototype_pred_points.csv"
        plot_tsne(pred_points, pred_records, "pred", fig_path, class_names)
        write_points_csv(csv_path, pred_points, pred_records)
        results.append({
            "source": "pred",
            "count": len(pred_records),
            "figure": str(fig_path),
            "csv": str(csv_path),
        })
        print("Redrew pred t-SNE figure to {}".format(os.path.abspath(fig_path)))

    meta_path = output_dir / "tsne_meta_redraw.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "class_names": class_names,
            "results": results,
            "prototype_counts": {
                "gt": len(gt_records),
                "pred": len(pred_records),
            },
            "source": "points",
        }, f, indent=2, ensure_ascii=False)
    print("Saved redraw metadata to {}".format(os.path.abspath(meta_path)))


def redraw_from_cache(output_dir, args):
    gt_cache = output_dir / "prototypes_gt.npz"
    pred_cache = output_dir / "prototypes_pred.npz"
    gt_records = load_prototype_cache(gt_cache)
    pred_records = load_prototype_cache(pred_cache)
    all_records = gt_records + pred_records

    if not all_records:
        raise RuntimeError(
            "No cached prototypes found in {}. Expected prototypes_gt.npz or prototypes_pred.npz.".format(
                output_dir
            )
        )

    num_classes = max(int(record["class_id"]) for record in all_records) + 1
    class_names = resolve_class_names(args.class_names, args.class_file, num_classes)
    apply_class_names_to_records(all_records, class_names)

    results = []
    gt_result = process_source(gt_records, "gt", output_dir, args, class_names)
    if gt_result:
        results.append(gt_result)
    pred_result = process_source(pred_records, "pred", output_dir, args, class_names)
    if pred_result:
        results.append(pred_result)

    meta_path = output_dir / "tsne_meta_redraw.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "class_names": class_names,
            "results": results,
            "prototype_counts": {
                "gt": len(gt_records),
                "pred": len(pred_records),
            },
            "source": "cache",
        }, f, indent=2, ensure_ascii=False)
    print("Redrew t-SNE figures from cache in {}".format(os.path.abspath(output_dir)))
    print("Saved redraw metadata to {}".format(os.path.abspath(meta_path)))


def process_source(source_records, source, output_dir, args, class_names):
    if not source_records:
        print("No {} prototypes found; skip t-SNE.".format(source))
        return None

    features = align_feature_dimensions(source_records, args.pca_dim, args.seed)
    points = run_tsne(features, args)

    fig_path = output_dir / "tsne_{}_by_class.png".format(source)
    csv_path = output_dir / "prototype_{}_points.csv".format(source)
    plot_tsne(points, source_records, source, fig_path, class_names)
    write_points_csv(csv_path, points, source_records)
    print("Saved {} t-SNE figure to {}".format(source, os.path.abspath(fig_path)))
    print("Saved {} t-SNE points to {}".format(source, os.path.abspath(csv_path)))
    return {
        "source": source,
        "count": len(source_records),
        "figure": str(fig_path),
        "csv": str(csv_path),
    }


def main():
    args = parse_args()

    global np
    global plt
    global PCA
    global TSNE
    global cv2
    global paddle
    global Config
    global config_check
    global get_sys_env
    global utils

    import numpy as np
    import matplotlib.pyplot as plt

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.redraw_from_points:
        redraw_from_points(output_dir, args)
        return

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    if args.redraw_from_cache:
        redraw_from_cache(output_dir, args)
        return

    import cv2
    import paddle
    from paddleseg.cvlibs import Config
    from paddleseg.utils import config_check, get_sys_env, utils

    paddle.set_device(choose_device(args.device))

    specs = load_model_specs(args)

    all_records = []
    model_meta = []
    for spec_index, spec in enumerate(specs):
        spec.setdefault("feature_layer", args.feature_layer)
        if args.list_layers and spec_index == 0:
            cfg = Config(spec["config"])
            apply_data_format(cfg, args.data_format)
            model = cfg.model
            for name, layer in iter_named_sublayers(model):
                print("{}\t{}".format(name, layer.__class__.__name__))
            return

        records, meta = extract_model_prototypes(spec, args)
        all_records.extend(records)
        model_meta.append(meta)

    all_records = cap_points_per_class(
        all_records, args.max_points_per_class, args.seed
    )
    gt_records = [record for record in all_records if record["source"] == "gt"]
    pred_records = [record for record in all_records if record["source"] == "pred"]

    if not all_records:
        raise RuntimeError("No prototypes were extracted. Try lowering --min_pixels.")

    num_classes = max(meta["num_classes"] for meta in model_meta)
    class_names = resolve_class_names(args.class_names, args.class_file, num_classes)
    apply_class_names_to_records(all_records, class_names)

    gt_cache = output_dir / "prototypes_gt.npz"
    pred_cache = output_dir / "prototypes_pred.npz"
    if gt_records:
        save_prototype_cache(gt_cache, gt_records)
    if pred_records:
        save_prototype_cache(pred_cache, pred_records)

    results = []
    gt_result = process_source(gt_records, "gt", output_dir, args, class_names)
    if gt_result:
        results.append(gt_result)
    pred_result = process_source(pred_records, "pred", output_dir, args, class_names)
    if pred_result:
        results.append(pred_result)

    meta_path = output_dir / "tsne_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump({
            "models": model_meta,
            "args": vars(args),
            "class_names": class_names,
            "results": results,
            "prototype_counts": {
                "gt": len(gt_records),
                "pred": len(pred_records),
            },
        }, f, indent=2, ensure_ascii=False)
    print("Saved metadata to {}".format(os.path.abspath(meta_path)))


if __name__ == "__main__":
    main()
