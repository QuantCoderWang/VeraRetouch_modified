# -*- coding: utf-8 -*-
"""探针2：区分『变糊（空间细节丢失）』与『只是颜色被调整（正常修图）』。
用法: python tests/_quality_probe2.py <input> <output>
"""
import sys

import numpy as np
from PIL import Image


def load(p):
    return np.asarray(Image.open(p).convert("L"), dtype=np.float64)


def lap_var(a):
    # 简易 Laplacian 能量：衡量锐度/细节总量
    kern = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(a, (3, 3))
    lap = np.abs(np.einsum("ijkl,kl->ij", win, kern))
    return float((lap ** 2).mean())


def main():
    src, out = sys.argv[1], sys.argv[2]
    a = load(src)
    b = load(out)
    la, lb = lap_var(a), lap_var(b)
    print(f"Laplacian 锐度  输入={la:.1f}  输出={lb:.1f}  比值={lb/la:.3f}")
    print("  (>0.98 → 输出没有比输入更糊；明显<0.8 → 发生了空间模糊/重采样)")

    # 全局单调色彩曲线拟合残差：修图=RGB→输出 是逐像素映射，理论上残差=纯噪声
    ca = np.asarray(Image.open(src).convert("RGB"), dtype=np.float64).reshape(-1, 3)
    cb = np.asarray(Image.open(out).convert("RGB"), dtype=np.float64).reshape(-1, 3)
    idx = np.random.default_rng(0).choice(len(ca), 200000, replace=False)
    x, y = ca[idx], cb[idx]
    xu = x / 255.0
    feats = np.stack([xu[:, c] ** p for c in range(3) for p in (1, 2, 3)] + [np.ones(len(xu))], axis=1)
    coef, res, *_ = np.linalg.lstsq(feats, y / 255.0, rcond=None)
    pred = feats @ coef
    psnr_res = 10 * np.log10(1.0 / max(np.mean((pred - y / 255.0) ** 2), 1e-12))
    psnr_chan = []
    for c in range(3):
        f = np.stack([xu[:, c] ** p for p in (1, 2, 3)] + [np.ones(len(xu))], axis=1)
        co, *_ = np.linalg.lstsq(f, y[:, c] / 255.0, rcond=None)
        d = f @ co - y[:, c] / 255.0
        psnr_chan.append(10 * np.log10(1.0 / max(np.mean(d ** 2), 1e-12)))
    print(f"RGB 联合多项式(3阶)全局映射残差 PSNR = {psnr_res:.2f} dB")
    print(f"  各通道(RGB)全局曲线残差 PSNR      = {psnr_chan[0]:.2f} / {psnr_chan[1]:.2f} / {psnr_chan[2]:.2f} dB")
    print("  (残差 PSNR >45dB → 输出≈输入的全局色彩曲线，无空间劣化；<38dB → 存在像素级局部差异)")


if __name__ == "__main__":
    main()
