import os
import sys

import h5py
import numpy as np


def summarize(path):
    print(f"--- {os.path.basename(path)}")
    with h5py.File(path, "r") as f:
        for key in f.keys():
            ds = f[key]
            print(f"{key}: shape={ds.shape}, dtype={ds.dtype}, chunks={ds.chunks}")
            if key == "label":
                arr = ds[()]
                vals = np.unique(arr)
                print(f"  unique={vals.tolist()}")
            else:
                sample = ds[(slice(None),) + tuple(slice(0, min(16, s)) for s in ds.shape[1:])] if ds.ndim == 3 else ds[()]
                print(f"  sample min={np.nanmin(sample)}, max={np.nanmax(sample)}")


def main():
    src_dir = sys.argv[1]
    for name in ["beijing_label.mat", "wuhan_label.mat", "beijing.mat", "wuhan.mat"]:
        summarize(os.path.join(src_dir, name))


if __name__ == "__main__":
    main()
