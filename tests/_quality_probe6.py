# -*- coding: utf-8 -*-
"""探针6：检查 chunk 接缝伪影（行差分周期性尖峰）。
用法: python tests/_quality_probe6.py <output_img>
"""
import sys

import numpy as np
from PIL import Image


def main():
    out = sys.argv[1]
    g = np.asarray(Image.open(out).convert("L"), dtype=np.float64)
    h, w = g.shape
    row_diff = np.abs(np.diff(g, axis=1)).mean(axis=1)     # 每行的横向差分能量（行内一致性参考）
    seam = np.abs(np.diff(g, axis=0)).mean(axis=1)        # 相邻行差：接缝处会跳变
    ratio = seam / np.maximum(row_diff[:-1], 1e-6)
    chunk = 262144
    chunk_rows = chunk / w
    print(f"图像 {w}x{h}，chunk={chunk} → 每 {chunk_rows:.2f} 行一个分块边界")
    # 找可疑接缝行（比值 > 中位数*5）
    med = np.median(ratio)
    cand = np.where(ratio > med * 5)[0]
    print(f"行差分中位比值 {med:.2f}；可疑接缝行 {len(cand)} 个（前20）：{cand[:20].tolist()}")
    # 检查这些行是否落在 chunk 边界附近（±2 行）
    boundaries = np.arange(1, h) [::1]
    hits = 0
    for r in cand:
        frac_row = r
        for k in range(int(h / chunk_rows) + 1):
            if abs(frac_row - k * chunk_rows) < 2.5:
                hits += 1
                print(f"  行 {r} 落在第 {k} 个 chunk 边界（{k*chunk_rows:.1f}）附近")
                break
    print(f"chunk 边界命中 {hits}/{len(cand)}")


if __name__ == "__main__":
    main()
