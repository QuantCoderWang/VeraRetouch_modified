# -*- coding: utf-8 -*-
"""嵌入式 runtime 构建验证：五页签 UI 构建 + 常规格式扫描。

用法（在各 app 目录下，用该目录自己的 runtime\python.exe 运行）：
    runtime\python.exe path\to\tests\_runtime_ui_check.py
脚本把当前工作目录（app 目录）加入 sys.path 后 import app。
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.getcwd())

from app import IMG_EXTS, ReferenceRetouchRunner, VeraRetouchRunner, build_ui, scan_images  # noqa: E402

args = argparse.Namespace(
    dtype="auto", model_path="./checkpoints/VeraRetouch",
    config_add_path="./configs/infer_config.yaml",
    pretrained_path="./checkpoints/encoder_renderer.pth",
    model_name="RetouchRenderer_Resnet18Encoder_InputCatMixedCBAM",
    temperature=0.2, top_p=0.8, num_beams=1, max_new_tokens=4096,
    chunk=262144, output_dir="./gradio_outputs",
)
runner = VeraRetouchRunner(args)
ref_runner = ReferenceRetouchRunner(args)
demo = build_ui(runner, ref_runner, args)

sliders = [c for c in demo.blocks.values() if type(c).__name__ == "Slider"]
buttons = [c for c in demo.blocks.values() if type(c).__name__ == "Button"]
dropdowns = [c for c in demo.blocks.values() if type(c).__name__ == "Dropdown"]
tabs = [c for c in demo.blocks.values() if type(c).__name__ == "Tab"]
assert len(sliders) == 38, len(sliders)
assert len(tabs) == 5, len(tabs)
assert len(dropdowns) == 9, len(dropdowns)
texts = [b.get_config().get("value") for b in buttons]
assert texts.count("开始批量修图") == 1

# 扫描入口冒烟：常规格式应被 scan_images 收进、RAW 不再收纳（2026-11-06 回退 RAW 支持）
with tempfile.TemporaryDirectory() as tmp:
    for name in ["a.jpg", "b.png"]:
        (Path(tmp) / name).write_bytes(b"x")
    (Path(tmp) / "c.cr2").write_bytes(b"x")
    names = [p.name for p in scan_images(tmp)]
    assert names == ["a.jpg", "b.png"], names
    assert ".cr2" not in IMG_EXTS and ".dng" not in IMG_EXTS

print(f"[OK] build_ui: {len(sliders)} sliders / {len(dropdowns)} dropdowns / "
      f"{len(tabs)} tabs；IMG_EXTS={len(IMG_EXTS)} 种常规格式、扫描 OK（无 RAW）")
