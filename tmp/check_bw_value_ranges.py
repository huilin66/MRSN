from pathlib import Path

import numpy as np
import tifffile as tif


ROOT = Path(r"\\158.132.186.40\isds\huilin\bdd\cp_data\C2Seg\src")
DS = ROOT / "C2Seg_BW"


def stats(arr):
    arr = np.asarray(arr)
    return {
        "dtype": str(arr.dtype),
        "shape": tuple(arr.shape),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
        "median": float(np.nanmedian(arr)),
    }


def print_stats(name, arr):
    s = stats(arr)
    print(
        f"{name}: dtype={s['dtype']} shape={s['shape']} "
        f"min={s['min']:.6g} max={s['max']:.6g} mean={s['mean']:.6g} median={s['median']:.6g}"
    )


def small_patch_path(modality, sample_id):
    for split in ("train", ""):
        path = DS / split / modality / f"{sample_id}.tiff" if split else DS / modality / f"{sample_id}.tiff"
        if path.is_file():
            return path
    raise FileNotFoundError(modality, sample_id)


ids = []
for line in (DS / "train.txt").read_text(encoding="utf-8").splitlines()[:10]:
    ids.append(Path(line.split()[0]).stem)

print("Small train patches, first 10 samples")
for modality in ("msi", "sar", "hsi"):
    values = []
    for sample_id in ids:
        values.append(tif.imread(small_patch_path(modality, sample_id)).ravel())
    print_stats(modality, np.concatenate(values))

print("\nFull wuhan tif, strided sample")
for modality in ("MSI", "SAR", "HSI"):
    path = ROOT / "tif_BW" / f"wuhan_{modality}.tif"
    image = tif.memmap(path)
    step = 16 if modality != "HSI" else 8
    if image.ndim == 3 and image.shape[0] < 300:
        sample = image[:, ::step, ::step]
    else:
        sample = image[::step, ::step, ...]
    print_stats(modality, sample)
