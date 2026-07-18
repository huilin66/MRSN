import pathlib
import sys

import tifffile


def main():
    tif_dir = pathlib.Path(sys.argv[1])
    names = [
        "beijing_label.tif",
        "beijing_label_color.tif",
        "beijing_MSI.tif",
        "beijing_HSI.tif",
        "wuhan_label_color.tif",
        "wuhan_HSI.tif",
    ]
    for name in names:
        arr = tifffile.memmap(tif_dir / name)
        print(name, arr.shape, arr.dtype, arr.size)
        del arr


if __name__ == "__main__":
    main()
