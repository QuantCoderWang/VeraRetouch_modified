# -*- coding: utf-8 -*-
"""探针7：暗区噪声放大归因——平滑映射的斜率是否能解释输出噪声的 std。
用法: python tests/_quality_probe7.py <input> <output>
原理：逐像素确定性映射下，均匀小块 std(输出) ≈ |曲线斜率| × std(输入)。
若实测 std 增益 >> 局部斜率 → 映射自身引入噪声（精度问题）；否则=正常放大。
"""
import sys

import numpy as np
from PIL import Image


def main():
    src, out = sys.argv[1], sys.argv[2]
    a = np.asarray(Image.open(src).convert("RGB"), dtype=np.float64)
    b = np.asarray(Image.open(out).convert("RGB"), dtype=np.float64)
    h, w = a.shape[:2]
    s = 96
    gray = a.mean(axis=2)
    # 挑一个「输入较暗但均匀」的小块（亮度 20-200 且 std 最小，排除过曝天空）
    best = None
    for y in range(0, h - s, 64):
        for x in range(0, w - s, 64):
            patch = gray[y:y + s, x:x + s]
            m = patch.mean()
            if not (20 <= m <= 80):
                continue
            score = patch.std()
            if best is None or score < best[0]:
                best = (score, y, x)
    _, y, x = best
    pa = a[y:y + s, x:x + s]
    pb = b[y:y + s, x:x + s]
    for c, ch in enumerate("RGB"):
        sa, sb = pa[..., c].std(), pb[..., c].std()
        ma = pa[..., c].mean()
        # 用整图数据估该亮度附近映射斜率（局部线性拟合）
        aa = a[..., c].ravel()
        bb = b[..., c].ravel()
        idx = np.abs(aa - ma) < 3
        if idx.sum() > 1000:
            slope = np.polyfit(aa[idx], bb[idx], 1)[0]
        else:
            slope = float("nan")
        print(f"{ch} 通道 @({x},{y}): 输入std={sa:.2f} 输出std={sb:.2f} 增益={sb/max(sa,1e-6):.2f}x "
              f"该亮度局部斜率={slope:.2f}")
    print("判定: 增益≈斜率 → 纯曲线放大（模型行为）；增益>>斜率 → 渲染自身加噪")
    print(f"暗区块位置: x={x}, y={y}, 亮度均值={gray[y:y+s, x:x+s].mean():.1f}")


if __name__ == "__main__":
    main()
