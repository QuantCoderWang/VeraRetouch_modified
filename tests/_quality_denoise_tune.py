# -*- coding: utf-8 -*-
"""降噪参数扫描：在真实输出图的暗区测降噪率与边缘保留。
用法: python tests/_quality_denoise_tune.py <output_img>
"""
import sys

import cv2
import numpy as np
from PIL import Image


def main():
    img = np.asarray(Image.open(sys.argv[1]).convert("RGB"))[:, :, ::-1]
    h, w = img.shape[:2]
    # 夹克暗区（上文 crop2 位置 x=1792,y=3328）取 512x512 做噪区测试
    y0, x0 = 3328, 1792
    patch = img[y0:y0 + 512, x0:x0 + 512]
    # 边缘保留测试：取树枝区（x=384,y=2688）
    edge = img[2688:2688 + 512, 384:384 + 512]
    ycc = cv2.cvtColor(edge, cv2.COLOR_BGR2YCrCb)
    cr_edge = ycc[..., 1].astype(np.float64)
    edge_energy = np.abs(np.diff(cr_edge)).mean()

    print(f"边缘区 Cr 梯度能量（越高越锐利）: 原={edge_energy:.2f}")
    print(f"{'参数':<38} {'暗区Cr std':<12} {'降噪率':<8} {'边缘保留%':<9}")
    for d, sc, ss in [(7, 12, 7), (9, 16, 7), (9, 20, 7), (11, 20, 7), (9, 26, 9), (13, 26, 9)]:
        yccp = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
        cr = cv2.bilateralFilter(yccp[..., 1], d, sc, ss)
        cb = cv2.bilateralFilter(yccp[..., 2], d, sc, ss)
        std_after = np.concatenate([cr.ravel(), cb.ravel()]).astype(np.float64).std()
        # 边缘保留：处理后 Cr 梯度能量 / 原
        yccE = cv2.cvtColor(edge, cv2.COLOR_BGR2YCrCb)
        crF = cv2.bilateralFilter(yccE[..., 1], d, sc, ss)
        keep = np.abs(np.diff(crF.astype(np.float64))).mean() / edge_energy * 100
        yccn = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
        base = np.concatenate([yccn[..., 1].ravel(), yccn[..., 2].ravel()]).astype(np.float64).std()
        print(f"d={d:<3} sigmaColor={sc:<3} sigmaSpace={ss:<3}      "
              f"{base:.1f}→{std_after:.1f}   {(1-std_after/base)*100:>5.1f}%   {keep:>5.1f}%")


if __name__ == "__main__":
    main()
