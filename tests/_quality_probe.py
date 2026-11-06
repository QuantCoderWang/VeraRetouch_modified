# -*- coding: utf-8 -*-
"""画质排查探针：用真实输入/输出对分离『保存环节』与『模型渲染环节』的损失。
用法: python tests/_quality_probe.py <input_path> <output_path>
"""
import io
import sys

import numpy as np
from PIL import Image


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def main():
    src, out = sys.argv[1], sys.argv[2]
    a = load(src)
    b = load(out)
    print(f"input  size : {a.shape[1]}x{a.shape[0]}")
    print(f"output size : {b.shape[1]}x{b.shape[0]}"
          + ("  (同分辨率)" if a.shape[:2] == b.shape[:2] else "  RESIZED!!"))
    print(f"PSNR input-vs-实际输出        : {psnr(a, b):.2f} dB")

    img = Image.open(src).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95)                      # 复刻 save_result 的 q95 默认(4:2:0)
    r95 = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float64)
    print(f"PSNR input-vs-q95重编码(不含模型): {psnr(a, r95):.2f} dB   文件 {len(buf.getvalue())//1024} KB")

    buf2 = io.BytesIO()
    img.save(buf2, "JPEG", quality=98, subsampling=0)      # q98 + 4:4:4 无色度降采样
    r98 = np.asarray(Image.open(buf2).convert("RGB"), dtype=np.float64)
    print(f"PSNR input-vs-q98+4:4:4重编码   : {psnr(a, r98):.2f} dB   文件 {len(buf2.getvalue())//1024} KB")

    buf3 = io.BytesIO()
    img.save(buf3, "PNG")
    print(f"PNG 无损重编码                  : inf dB          文件 {len(buf3.getvalue())//1024} KB")


if __name__ == "__main__":
    main()
