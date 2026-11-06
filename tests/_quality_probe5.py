# -*- coding: utf-8 -*-
"""探针5：自动挑最丰富细节区，生成 100% 对比图（左输入右输出）。
用法: python tests/_quality_probe5.py <input> <output> <save_path>
"""
import sys

import numpy as np
from PIL import Image


def main():
    src, out, save = sys.argv[1], sys.argv[2], sys.argv[3]
    a = np.asarray(Image.open(src).convert("RGB"), dtype=np.uint8)
    b = np.asarray(Image.open(out).convert("RGB"), dtype=np.uint8)
    h, w = a.shape[:2]
    s = 384
    # 滑窗挑方差最高的两块（步长 128）
    gray = a.mean(axis=2)
    best = []
    for y in range(0, h - s, 128):
        for x in range(0, w - s, 128):
            v = gray[y:y + s, x:x + s].var()
            best.append((v, y, x))
    best.sort(reverse=True)
    picks = [(best[0][1], best[0][2]), (best[len(best) // 2][1], best[len(best) // 2][2])
             if len(best) > 3 else (best[-1][1], best[-1][2])]
    gap = 10
    canvas = np.full((2 * s + gap * 3, 2 * s + gap, 3), 128, dtype=np.uint8)
    for row, (y, x) in enumerate(picks):
        ca = a[y:y + s, x:x + s]
        cb = b[y:y + s, x:x + s]
        r0 = row * (s + gap) + gap
        canvas[r0:r0 + s, :s] = ca
        canvas[r0:r0 + s, s + gap:] = cb
        print(f"crop{row+1} 位置 x={x} y={y}")
    Image.fromarray(canvas).save(save)
    print("saved:", save)


if __name__ == "__main__":
    main()
