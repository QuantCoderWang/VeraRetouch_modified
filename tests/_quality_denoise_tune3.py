# -*- coding: utf-8 -*-
"""亮度降噪参数搜索：NLM h 与 guided 对比（真实照片）。
用法: python tests/_quality_denoise_tune3.py <output_img>
"""
import sys
import time

import cv2
import numpy as np
from PIL import Image


def hf_std(plane, k=5):
    p = plane.astype(np.float64)
    return float((p - cv2.GaussianBlur(plane, (k, k), 0).astype(np.float64)).std())


def main():
    img = np.asarray(Image.open(sys.argv[1]).convert("RGB"))[:, :, ::-1]
    patch = img[3328:3840, 1792:2304]          # 夹克暗区（噪）
    edge = img[2688:3200, 384:896]             # 树枝区（锐利纹理）
    Y0 = cv2.cvtColor(edge, cv2.COLOR_BGR2YCrCb)[..., 0]
    edge0 = np.abs(np.diff(Y0.astype(np.float64))).mean()
    noise0 = hf_std(cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)[..., 0])
    print(f"暗区亮度高频噪声={noise0:.2f}  树枝边缘梯度能量={edge0:.2f}")

    for hval in (6, 10, 14):
        t = time.perf_counter()
        Yp = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)[..., 0]
        Yd = cv2.fastNlMeansDenoising(Yp, None, hval, 7, 21)
        n = hf_std(Yd)
        Ye = cv2.cvtColor(edge, cv2.COLOR_BGR2YCrCb)[..., 0]
        Yde = cv2.fastNlMeansDenoising(Ye, None, hval, 7, 21)
        keep = np.abs(np.diff(Yde.astype(np.float64))).mean() / edge0 * 100
        t = time.perf_counter() - t
        print(f"NLM h={hval:<3} 噪声 {noise0:.2f}→{n:.2f} ({(1-n/noise0)*100:.0f}%)  "
              f"边缘保留 {keep:.0f}%  耗时 {t*1000:.0f}ms/0.25MP (全图约 ×10-12)")


if __name__ == "__main__":
    main()
