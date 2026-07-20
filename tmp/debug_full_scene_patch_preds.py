import argparse
import os
from pathlib import Path

import numpy as np
import paddle

REPO_ROOT = Path(__file__).resolve().parents[1]
PADDLECD_ROOT = REPO_ROOT / "PaddleCD"
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PADDLECD_ROOT) not in sys.path:
    sys.path.insert(0, str(PADDLECD_ROOT))

from tools import infer_full_scene as full

from paddleseg.cvlibs import Config
from paddleseg.utils import utils as seg_utils


def pred_stats(logits):
    pred = np.argmax(logits, axis=1).astype("int32")
    vals, counts = np.unique(pred, return_counts=True)
    return dict(zip(vals.tolist(), counts.tolist()))


def logit_stats(logits):
    return {
        "shape": tuple(logits.shape),
        "min": float(logits.min()),
        "max": float(logits.max()),
        "mean": float(logits.mean()),
        "std": float(logits.std()),
        "pred": pred_stats(logits),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--dataset", default="BW")
    parser.add_argument("--scene", default="wuhan")
    parser.add_argument("--data_root", required=True)
    args = parser.parse_args()

    cfg = Config(str(REPO_ROOT / "PaddleCD" / "c2seg_config" / f"{args.model_name}.yml"))
    model = cfg.model
    seg_utils.load_entire_model(model, str(REPO_ROOT / "output" / args.model_name / "best_model" / "model.pdparams"))
    model.eval()

    paddle.set_device("gpu")
    val_ds = cfg.val_dataset
    im1, im2, label = val_ds[0]
    with paddle.no_grad():
        logits = full.forward_logits(
            model,
            paddle.to_tensor(im1[None, :, :, :].astype("float32")),
            paddle.to_tensor(im2[None, :, :, :].astype("float32")),
            paddle,
        ).numpy()
    print("small_val", logit_stats(logits), "label_unique", np.unique(label).tolist())
    print("small_val im1", im1.shape, float(im1.min()), float(im1.max()), float(im1.mean()), float(im1.std()))
    print("small_val im2", im2.shape, float(im2.min()), float(im2.max()), float(im2.mean()), float(im2.std()))

    source_args = argparse.Namespace(
        dataset=args.dataset, scene=args.scene, data_root=args.data_root,
        scene_root="", msi="", sar="", hsi="", mat="",
    )
    source = full.resolve_source(source_args)
    mean1, std1, mean2, std2 = full.get_normalize_params(cfg)
    msi_bands = source.msi.resolve_bands(None)
    sar_bands = source.sar.resolve_bands(None)
    hsi_bands = source.hsi.resolve_bands(None)
    if len(hsi_bands) > len(mean2):
        hsi_bands = hsi_bands[:len(mean2)]
    im1, im2 = source.read_patch(0, 0, 256, 256, msi_bands, sar_bands, hsi_bands)
    raw1, raw2 = im1.copy(), im2.copy()
    im2 = full.maybe_rescale_hsi(im2, mean2, None)
    im1, im2 = full.normalize_inputs(im1, im2, mean1, std1, mean2, std2)
    with paddle.no_grad():
        logits = full.forward_logits(
            model,
            paddle.to_tensor(im1[None, :, :, :]),
            paddle.to_tensor(im2[None, :, :, :]),
            paddle,
        ).numpy()
    print("full_patch", logit_stats(logits))
    print("full_raw im1", raw1.shape, float(raw1.min()), float(raw1.max()), float(raw1.mean()), float(raw1.std()))
    print("full_raw im2", raw2.shape, float(raw2.min()), float(raw2.max()), float(raw2.mean()), float(raw2.std()))
    print("full_norm im1", im1.shape, float(im1.min()), float(im1.max()), float(im1.mean()), float(im1.std()))
    print("full_norm im2", im2.shape, float(im2.min()), float(im2.max()), float(im2.mean()), float(im2.std()))
    source.close()


if __name__ == "__main__":
    main()
