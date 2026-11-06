# -*- coding: utf-8 -*-
"""探针4：全部真实 input/output 对的体检表（PSNR/锐度比/过曝/EXIF方向）。
用法: python tests/_quality_probe4.py <in_dir> <out_dir>
"""
import os
import sys

import numpy as np
from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


def psnr(a, b):
    mse = np.mean((np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def main():
    in_dir, out_dir = sys.argv[1], sys.argv[2]
    files = sorted(f for f in os.listdir(in_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    print(f"{'文件':<28} {'尺寸':<12} {'PSNR':<7} {'锐度比':<7} {'过曝率(入/出)':<16} 方向(入/出)")
    for name in files:
        src = os.path.join(in_dir, name)
        out = os.path.join(out_dir, os.path.splitext(name)[0] + os.path.splitext(name)[1].lower())
        if not os.path.exists(out):
            print(f"{name:<28} 输出缺失")
            continue
        ia = Image.open(src)
        ib = Image.open(out)
        sa = np.asarray(ia.convert("L"), dtype=np.float64)
        sb = np.asarray(ib.convert("L"), dtype=np.float64)
        sha = np.asarray(ia.convert("L"), dtype=np.float64)
        lap = lambda x: np.abs(np.diff(x.astype(np.float64), axis=0)).mean() + np.abs(np.diff(x.astype(np.float64), axis=1)).mean()
        a = np.asarray(ia.convert("RGB"), dtype=np.float64)
        b = np.asarray(ib.convert("RGB"), dtype=np.float64)
        hi_a = (a >= 253 / 255.0).all(-1).mean() * 100
        hi_b = (b >= 253 / 255.0).all(-1).mean() * 100
        ori_a = ia.getexif().get(274, 1)
        ori_b = ib.getexif().get(274, 1)
        p = psnr(a, b)
        r = lap(sb) / max(lap(sha), 1e-9)
        print(f"{name:<28} {str(a.shape[1])+'x'+str(a.shape[0]):<12} {p:<7.1f} {r:<7.2f} "
              f"{hi_a:.1f}% / {hi_b:.1f}%      {ori_a}/{ori_b}")


if __name__ == "__main__":
    main()
