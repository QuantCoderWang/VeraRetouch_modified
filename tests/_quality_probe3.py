# -*- coding: utf-8 -*-
"""探针3：检查高光溢出/暗部裁剪/色偏 + 生成 100% 对比图供肉眼检查。
用法: python tests/_quality_probe3.py <input> <output>
"""
import sys

import numpy as np
from PIL import Image


def main():
    src, out = sys.argv[1], sys.argv[2]
    a = np.asarray(Image.open(src).convert("RGB"), dtype=np.float64) / 255.0
    b = np.asarray(Image.open(out).convert("RGB"), dtype=np.float64) / 255.0
    for tag, im in (("输入", a), ("输出", b)):
        clip_hi = (im >= 253 / 255.0).all(axis=-1).mean() * 100
        clip_lo = (im <= 2 / 255.0).all(axis=-1).mean() * 100
        mean = im.mean(axis=(0, 1)) * 255
        print(f"{tag}: 过曝像素 {clip_hi:.2f}%  欠曝(纯黑)像素 {clip_lo:.2f}%  "
              f"均值 R{mean[0]:.1f} G{mean[1]:.1f} B{mean[2]:.1f}")

    # 100% 对比图：人物面部附近 + 天空渐变区各取一 crop，上下并排（左输入右输出）
    h, w = a.shape[:2]
    # 面部区域简单启发：取上 1/3 中间的 crop（人像拍摄习惯在画面中部）
    def crop(im, cy, cx, s=384):
        y0 = max(0, min(h - s, cy - s // 2))
        x0 = max(0, min(w - s, cx - s // 2))
        return im[y0:y0 + s, x0:x0 + s]
    face_a = crop(a, int(h * 0.42), int(w * 0.52))
    face_b = crop(b, int(h * 0.42), int(w * 0.52))
    sky_a = crop(a, int(h * 0.10), int(w * 0.30))
    sky_b = crop(b, int(h * 0.10), int(w * 0.30))
    gap = 8
    canvas = np.ones((384 * 2 + gap * 3, 384 * 2 + gap, 3)) * 128
    canvas[:384, :384] = face_a
    canvas[:384, 384 + gap:] = face_b
    canvas[384 + gap: 384 * 2 + gap, :384] = sky_a
    canvas[384 + gap: 384 * 2 + gap, 384 + gap:] = sky_b
    img = Image.fromarray((canvas).astype(np.uint8))
    p = "gradio_outputs/quality_compare_100pct.png"
    import os
    os.makedirs("gradio_outputs", exist_ok=True)
    img.save(p)
    print("对比图已保存:", p)


if __name__ == "__main__":
    main()
