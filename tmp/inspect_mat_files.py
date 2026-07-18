import os
import sys

from scipy.io import whosmat


def main():
    src_dir = sys.argv[1]
    for name in ["beijing_label.mat", "wuhan_label.mat", "beijing.mat", "wuhan.mat"]:
        path = os.path.join(src_dir, name)
        print(f"--- {name}")
        try:
            for item in whosmat(path):
                print(item)
        except Exception as exc:
            print(type(exc).__name__, exc)


if __name__ == "__main__":
    main()
