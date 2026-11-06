# -*- coding: utf-8 -*-
"""细节回贴降噪四档对比图（真实照片）：原图 | 轻α0.6 | 中α0.35 | 强α0.15 | 旧裸h10。
用法: python tests/_quality_denoise_visual2.py <output_img> <save_path>
"""
import sys

import cv2
import numpy as np
from PIL import Image


def dn(img, alpha, h=10):
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y_orig = ycc[..., 0].astype(np.float32)
    y_dn = cv2.fastNlMeansDenoising(y_orig.astype(np.uint8), None, h, 7, 21)
    restored = y_dn.astype(np.float32) + alpha * (y_orig - y_dn.astype(np.float32))
    ycc[..., 0] = np.clip(restored + 0.5, 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def main():
    src, save = sys.argv[1], sys.argv[2]
    img = np.asarray(Image.open(src).convert("RGB"))[:, :, ::-1]
    s = 300
    regions = [(3328, 1792, "噪区"), (2688, 384, "纹理区")]
    rows = []
    for y0, x0, _ in regions:
        strip = []
        p = img[y0:y0 + s, x0:x0 + s]
        variants = [None, 0.6, 0.35, 0.15, -1]  # -1 = 旧裸 h10
        for v in variants:
            if v is None:
                out = p
            elif v == -1:
                out = dn(p, 0.0)
            else:
                out = dn(p, v)
            strip.append(out[:, :, ::-1])
        rows.append(np.concatenate(strip, axis=1))
    gap = np.full((6, rows[0].shape[1], 3), 255, dtype=np.uint8)
    canvas = np.concatenate([rows[0], gap, rows[1]], axis=0)
    Image.fromarray(canvas).save(save)
    print("saved:", save, "（每行：原图 | 轻0.6 | 中0.35 | 强0.15 | 旧裸h10；上噪区下纹理区）")


if __name__ == "__main__":
    main()
