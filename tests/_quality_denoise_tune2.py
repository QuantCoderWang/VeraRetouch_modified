# -*- coding: utf-8 -*-
"""降噪方法对比：高斯 vs 双边 vs NLM（色度通道 / 真实照片）。
用法: python tests/_quality_denoise_tune2.py <output_img>
"""
import sys
import time

import cv2
import numpy as np
from PIL import Image


def chroma_dn(img, fn):
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycc[..., 1], ycc[..., 2] = fn(ycc[..., 1]), fn(ycc[..., 2])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def main():
    img = np.asarray(Image.open(sys.argv[1]).convert("RGB"))[:, :, ::-1]
    patch = img[3328:3328 + 512, 1792:1792 + 512]      # 夹克暗区（噪）
    edge = img[2688:2688 + 512, 384:384 + 512]         # 树枝区（锐）
    cr0 = cv2.cvtColor(edge, cv2.COLOR_BGR2YCrCb)[..., 1].astype(np.float64)
    edge0 = np.abs(np.diff(cr0)).mean()
    noise0 = np.concatenate([cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)[..., 1].ravel(),
                             cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)[..., 2].ravel()]).std()
    print(f"暗区色噪(std 参考)={noise0:.1f}  边缘区 Cr 梯度能量={edge0:.2f}   高频彩噪见各行")
    cands = {
        "高斯 k=3": lambda c: cv2.GaussianBlur(c, (3, 3), 0),
        "高斯 k=5": lambda c: cv2.GaussianBlur(c, (5, 5), 0),
        "高斯 k=7": lambda c: cv2.GaussianBlur(c, (7, 7), 0),
        "中值 k=3": lambda c: cv2.medianBlur(c, 3),
        "中值 k=5": lambda c: cv2.medianBlur(c, 5),
        "NLM 强度10": None,   # 特殊处理
    }

    def noise_std(ycc):
        # 高频残差（像素-局部均值）：直接对应可见彩斑强度
        vals = []
        for c in (1, 2):
            plane = ycc[..., c].astype(np.float64)
            hf = plane - cv2.GaussianBlur(plane, (5, 5), 0)
            vals.append(hf.ravel())
        return float(np.concatenate(vals).std())

    ycc_patch0 = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
    noise0 = noise_std(ycc_patch0)
    for name, fn in cands.items():
        t = time.perf_counter()
        if name == "NLM 强度10":
            out_patch = cv2.fastNlMeansDenoisingColored(patch, None, 10, 10, 7, 21)
            t0 = time.perf_counter() - t
            out_edge = cv2.fastNlMeansDenoisingColored(edge, None, 10, 10, 7, 21)
        else:
            out_patch = chroma_dn(patch, fn)
            t0 = time.perf_counter() - t
            out_edge = chroma_dn(edge, fn)
        cr = cv2.cvtColor(out_edge, cv2.COLOR_BGR2YCrCb)[..., 1].astype(np.float64)
        keep = np.abs(np.diff(cr)).mean() / edge0 * 100
        n = noise_std(cv2.cvtColor(out_patch, cv2.COLOR_BGR2YCrCb))
        scale = "  (全图约 ×12)" if "NLM" in name else ""
        print(f"{name:<12} 高频彩噪 {noise0:.1f}→{n:.1f} ({(1-n/noise0)*100:.0f}%)  "
              f"边缘保留 {keep:.0f}%  耗时 {t0*1000:.0f}ms/0.25MP{scale}")


if __name__ == "__main__":
    main()
