# -*- coding: utf-8 -*-
"""生成降噪前后的 100% 对比图（夹克噪区 + 树枝纹理区）。
用法: python tests/_quality_denoise_visual.py <output_img> <save_path>
"""
import sys
import time

import cv2
import numpy as np
from PIL import Image


def dn(img, hval):
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycc[..., 0] = cv2.fastNlMeansDenoising(ycc[..., 0], None, hval, 7, 21)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def main():
    src, save = sys.argv[1], sys.argv[2]
    img = np.asarray(Image.open(src).convert("RGB"))[:, :, ::-1]
    s = 320
    # 两个测试区：夹克暗区（噪）/ 树枝纹理区（锐）
    regions = [(3328, 1792, "噪区"), (2688, 384, "纹理区")]
    rows = []
    for y0, x0, _tag in regions:
        strip = []
        # 依次：原图、h=6、h=10
        for variant in [None, 6, 10]:
            t = time.perf_counter()
            p = img[y0:y0 + s, x0:x0 + s]
            if variant is None:
                out = p
            else:
                out = dn(p, variant)
            strip.append(out[:, :, ::-1])
        rows.append(np.concatenate(strip, axis=1))
    gap = np.full((6, rows[0].shape[1], 3), 255, dtype=np.uint8)
    canvas = np.concatenate([rows[0], gap, rows[1]], axis=0)
    Image.fromarray(canvas).save(save)
    print("saved:", save, "(每行：原图 | h=6 | h=10；上=暗区下=纹理区)")


if __name__ == "__main__":
    main()
