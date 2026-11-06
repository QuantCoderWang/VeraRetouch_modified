#
# Smoke test for the reworked UI — builds the Gradio Blocks WITHOUT loading
# any model weights, then validates:
#   1. param slider pack/parse round-trips (SLIDER_KEYS protocol order;
#      json_to_sliders/sliders_to_json remain internal conversion utilities
#      after the JSON expert mode was removed from the UI in 2026-11-05)
#   2. layout-order mapping used by the two-column param tab
#   3. full Blocks construction (all tabs, events, sliders)
#   4. device_util adaptive chunking logic
# Run:  vera_venv\Scripts\python tests\ui_smoke_test.py
#
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

import torch  # noqa: E402

from app import ReferenceRetouchRunner, VeraRetouchRunner, build_ui  # noqa: E402
from ui import i18n  # noqa: E402
from ui.params_def import (  # noqa: E402
    SLIDER_KEYS,
    default_slider_values,
    json_to_sliders,
    layout_order_keys,
    sliders_to_json,
    values_from_param_file,
)

DEVICE_UTIL = None


def test_params_roundtrip():
    n = len(SLIDER_KEYS)
    assert n == 38, f"expected 38 params, got {n}"

    values = [((-i * 7) % 201) - 100 for i in range(n)]
    js = sliders_to_json(*values)
    parsed, unknown = json_to_sliders(js)
    assert parsed == values, "sliders_to_json/json_to_sliders round-trip failed"
    assert unknown == []

    # missing keys default to 0, unknown keys reported
    js2 = f'{{"{SLIDER_KEYS[0]}": {{"value": 42}}}}'
    parsed2, unknown2 = json_to_sliders(js2)
    assert parsed2[0] == 42 and sum(parsed2[1:]) == 0
    assert unknown2 == []

    # out-of-range clamping
    data3 = '{"Exposure2012":{"value":999},"IncrementalTemperature":{"value":-999},"NoSuchKey":{"value":5}}'
    parsed3, unknown3 = json_to_sliders(data3)
    assert parsed3[0] == 100 and parsed3[SLIDER_KEYS.index("IncrementalTemperature")] == -100
    assert unknown3 == ["NoSuchKey"]

    # sample param file -> 38 values
    sample = values_from_param_file("data_samples/param.json")
    assert len(sample) == 38 and sample[0] == 30  # Exposure2012=30
    js4 = sliders_to_json(*sample)
    back, _ = json_to_sliders(js4)
    assert back == sample
    print("[OK] param round-trips")


def test_layout_mapping():
    lk = layout_order_keys()
    assert len(lk) == 38 and set(lk) == set(SLIDER_KEYS)
    # layout order must differ from protocol order (two-column split) unless trivial
    print("[OK] layout order covers all keys")


def test_build_ui():
    args = argparse.Namespace(
        dtype="auto", model_path="./checkpoints/VeraRetouch",
        config_add_path="./configs/infer_config.yaml",
        pretrained_path="./checkpoints/encoder_renderer.pth",
        model_name="RetouchRenderer_Resnet18Encoder_InputCatMixedCBAM",
        temperature=0.2, top_p=0.8, num_beams=1, max_new_tokens=4096,
        chunk=262144, output_dir="./gradio_outputs",
    )
    runner = VeraRetouchRunner(args)          # constructor only; model NOT loaded
    ref_runner = ReferenceRetouchRunner(args)
    demo = build_ui(runner, ref_runner, args)

    sliders = [c for c in demo.blocks.values() if type(c).__name__ == "Slider"]
    assert len(sliders) == 38, f"expected 38 sliders in built UI, got {len(sliders)}"
    buttons = [c for c in demo.blocks.values() if type(c).__name__ == "Button"]
    assert len(buttons) >= 10  # 4 start + reset/sample + 批量页{浏览×2,扫描,开始}
    radios = [c for c in demo.blocks.values() if type(c).__name__ == "Radio"]
    assert len(radios) == 0, "JSON 专家模式已移除，不应存在参数输入方式单选框"
    dropdowns = [c for c in demo.blocks.values() if type(c).__name__ == "Dropdown"]
    export_dropdowns = [d for d in dropdowns
                        if "auto" in [v for _, v in (d.get_config().get("choices") or [])]]
    denoise_dropdowns = [d for d in dropdowns
                         if d.get_config().get("label") == i18n.DENOISE_LABEL]
    assert len(export_dropdowns) == 4, f"expected 4 export-format dropdowns, got {len(export_dropdowns)}"
    assert len(denoise_dropdowns) == 5, f"expected 5 denoise dropdowns, got {len(denoise_dropdowns)}"
    for d in export_dropdowns:
        choice_values = [v for _, v in (d.choices or [])]
        assert "auto" in choice_values, "导出格式下拉框必须包含「跟随输入格式」(auto)"
    for d in denoise_dropdowns:
        vals = [v for _, v in (d.get_config().get("choices") or [])]
        assert "off" in vals and "light" in vals and "medium" in vals and "strong" in vals, \
            "输出降噪下拉框必须包含 off/light/medium/strong 四档"
    tabs = [c for c in demo.blocks.values() if type(c).__name__ == "Tab"]
    assert len(tabs) == 5, f"batch tab added 2026-11-05: expected 5 tabs, got {len(tabs)}"
    # gradio 6 将按钮标题存于 value（非 label）；页签标题在 label（get_config 读取）
    button_texts = [b.get_config().get("value") for b in buttons]
    assert button_texts.count("开始批量修图") == 1, "开始批量修图按钮必须有且只有一个"
    assert button_texts.count("浏览…") == 2, "批量页需要两个『浏览…』按钮（输入/输出文件夹）"
    assert button_texts.count("扫描图片") == 1, "批量页需要『扫描图片』按钮"
    tab_labels = [c.get_config().get("label") for c in tabs]
    assert any("批量自动修图" in (l or "") for l in tab_labels), "批量自动修图页签必须存在"
    print(f"[OK] Blocks built: {len(sliders)} sliders, {len(buttons)} buttons, "
          f"{len(dropdowns)} dropdowns, {len(tabs)} tabs (batch tab included)")


def test_save_result_formats():
    import os
    import tempfile
    from PIL import Image
    from app import save_result
    rgb = Image.new("RGB", (4, 4), (10, 20, 30))
    rgba = Image.new("RGBA", (4, 4), (10, 20, 30, 255))
    tmp = tempfile.mkdtemp(prefix="vr_save_result_")
    assert save_result(rgb, None, tmp, fmt_key="jpg").endswith(".jpg")
    assert save_result(rgba, None, tmp, fmt_key="jpg").endswith(".jpg")  # RGBA→JPG 自动转 RGB 不炸
    assert save_result(rgb, None, tmp, fmt_key="png").endswith(".png")
    assert save_result(rgb, None, tmp, fmt_key="webp").endswith(".webp")
    # auto: 跟随输入格式；识别不了的扩展名 / 无源路径回退 png
    assert save_result(rgb, None, tmp, fmt_key="auto", src_path="a.jpg").endswith(".jpg")
    assert save_result(rgb, None, tmp, fmt_key="auto", src_path="a.jpeg").endswith(".jpg")
    assert save_result(rgb, None, tmp, fmt_key="auto", src_path="a.webp").endswith(".webp")
    assert save_result(rgb, None, tmp, fmt_key="auto", src_path="a.bmp").endswith(".bmp")
    assert save_result(rgb, None, tmp, fmt_key="auto", src_path="a.xxx").endswith(".png")
    assert save_result(rgb, None, tmp, fmt_key="auto", src_path=None).endswith(".png")
    print("[OK] save_result format selection (auto follow / png / jpg / webp)")


def test_denoise_and_exif():
    """输出降噪（细节回贴式，方案一）+ EXIF 方向转正（2026-11-05/06 画质回归）。"""
    import cv2
    import numpy as np
    from PIL import Image
    from app import bake_exif_orientation, denoise_output

    # --- denoise_output：细节回贴式（final = dn + α×(orig−dn)）---
    # 在 Y 通道注入噪点（模拟"提亮放大传感器噪声"场景，仍是亮度-dominated）
    h = w = 128
    rr = np.random.default_rng(7)
    y_noisy = np.clip(50 + rr.normal(0, 8, (h, w)), 0, 255).astype(np.uint8)
    cr0 = np.full((h, w), 130, dtype=np.uint8)
    cb0 = np.full((h, w), 120, dtype=np.uint8)
    bgr = cv2.cvtColor(cv2.merge([y_noisy, cr0, cb0]), cv2.COLOR_YCrCb2BGR)
    src = Image.fromarray(bgr[:, :, ::-1])                     # BGR→RGB

    def hf_of(img_bgr):
        y = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float64)
        return float((y - cv2.GaussianBlur(y.astype(np.uint8), (5, 5), 0).astype(np.float64)).std())

    def y_hf_from_pil(pil_img, k=5):
        arr = np.asarray(pil_img.convert("RGB"))[:, :, ::-1]
        y = cv2.cvtColor(arr, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float64)
        return float((y - cv2.GaussianBlur(y.astype(np.uint8), (k, k), 0).astype(np.float64)).std())

    mid = denoise_output(src, alpha=0.35)     # 中档：均衡
    strong = denoise_output(src, alpha=0.15)  # 强档：最干净
    light = denoise_output(src, alpha=0.6)    # 轻档：细节优先
    noise_before = y_hf_from_pil(src)
    noise_mid = y_hf_from_pil(mid)
    noise_strong = y_hf_from_pil(strong)
    noise_light = y_hf_from_pil(light)
    # 中档明显降噪
    assert noise_mid < noise_before * 0.75, \
        f"中档应明显降噪: {noise_before:.2f} -> {noise_mid:.2f}"
    # 档位单调性：越强越干净
    assert noise_strong < noise_mid < noise_light, \
        (noise_light, noise_mid, noise_strong)
    # 色度必须不动（NLM+回贴只作用 Y）
    out_arr = np.asarray(mid.convert("RGB"))[:, :, ::-1]
    _y, _cr, _cb = cv2.split(cv2.cvtColor(out_arr.astype(np.uint8), cv2.COLOR_BGR2YCrCb))
    assert abs(float(_cr.astype(np.float64).mean()) - 130) < 1.5, "Cr 均值不应漂移"
    assert abs(float(_cb.astype(np.float64).mean()) - 120) < 1.5, "Cb 均值不应漂移"
    print(f"[OK] denoise_output: 亮度高频 {noise_before:.2f} → 轻 {noise_light:.2f} / "
          f"中 {noise_mid:.2f} / 强 {noise_strong:.2f}，色度保持 130/120")

    # --- bake_exif_orientation: 方向标签的源图，输出应被像素转正 ---
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "r8.jpg")
        im = Image.new("RGB", (200, 100), (10, 200, 30))
        exif = Image.Exif()
        exif[274] = 8   # ROTATE_90: 源图在相机里是横拍、看图软件会转 90°
        im.save(p, exif=exif)
        out_img = bake_exif_orientation(im.copy(), p)
        assert out_img.size == (100, 200), \
            f"方向=8 的输出应转正为 100x200，得到 {out_img.size}"
        # 方向=1：不转
        exif[274] = 1
        p1 = os.path.join(tmp, "r1.jpg")
        im.save(p1, exif=exif)
        assert bake_exif_orientation(im.copy(), p1).size == (200, 100)
        # 无 EXIF/无文件：原样
        assert bake_exif_orientation(im.copy(), None).size == (200, 100)
    print("[OK] bake_exif_orientation: 方向标签正确像素转正（不再横竖颠倒）")


def test_batch_helpers():
    """批量自动修图辅助函数纯逻辑断言（不加载模型，技术方案 §5-T3/T5）。"""
    import tempfile
    from pathlib import Path
    from app import _fmt_eta, _already_done, overall_progress, scan_images

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for name in ["a.jpg", "b.JPG", "c.png", "d.webp", "e.bmp",
                     "notes.txt", ".hidden.jpg"]:
            (d / name).write_bytes(b"x")
        (d / "sub").mkdir()
        (d / "sub" / "nested.jpg").write_bytes(b"x")
        names = [p.name for p in scan_images(tmp)]
        assert names == ["a.jpg", "b.JPG", "c.png", "d.webp", "e.bmp"], names
        assert scan_images(str(d / "not_exist")) is None
        assert scan_images("") is None
        # 断点续跑判断：输出目录已有同基名同扩展 → True
        out = d / "out"
        out.mkdir()
        assert not _already_done(d / "a.jpg", out)
        (out / "a.jpg").write_bytes(b"x")
        assert _already_done(d / "a.jpg", out)

    # overall_progress：第 1 张（无历史均值）→ 线性退化
    p, eta = overall_progress(1, 10, [], 0.0)
    assert p == 0.0 and eta is None
    # 3 张完成各 10s、第 4 张进行 5s → prog=(30+0.5×10)/100=0.35，eta=10×7
    p, eta = overall_progress(4, 10, [10.0] * 3, 5.0)
    assert abs(p - 0.35) < 1e-9, p
    assert abs(eta - 70.0) < 1e-9, eta
    # 第 10 张进行中 credit=0.5 → prog=0.95，恒不超 0.99（完成只认真实收尾）
    p, _ = overall_progress(10, 10, [10.0] * 9, 5.0)
    assert abs(p - 0.95) < 1e-9, p
    assert p <= 0.99
    assert _fmt_eta(None) == "统计中…"
    assert _fmt_eta(95) == "1分35秒"
    print("[OK] batch helpers: scan_images / overall_progress / _already_done / _fmt_eta")


def test_batch_loop_end_to_end():
    """batch_loop 生成器端到端（假 runner，不加载模型）：技术方案 §5-T4。

    回归背景：最初 batch_handler 用 `return batch_loop(...)` 把生成器对象当
    单个返回值，gradio 报 "didn't return enough output values (needed: 4,
    returned: 1)" —— 本测试明确断言产出是四元组流。
    """
    import tempfile
    from pathlib import Path
    from app import batch_loop

    class _FakeRunner:
        def run(self, mode, img_path, prompt=None, param_json=None, work_dir=None, progress=None):
            from PIL import Image
            return Image.new("RGB", (8, 8), (1, 2, 3)), "分析文本 for test"

    class _FakeProg:
        def __call__(self, *a, **k):
            pass

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in"
        out = Path(tmp) / "out"
        src.mkdir(); out.mkdir()
        for name in ["p1.jpg", "p2.png", "skipme.bmp"]:
            (src / name).write_bytes(b"fake")
        (out / "skipme.bmp").write_bytes(b"fake")   # ← 触发"输出已存在 → 跳过"

        gen = batch_loop(_FakeRunner(), str(src), str(out), _FakeProg(), _FakeProg())
        outputs = list(gen)
        assert outputs, "batch_loop 必须产出至少一个三元组"
        # 2026-11-05 精简后每帧 = (实时统计行, 汇总文本, 按钮状态) 三元组
        # （页面上的预览图/最近一张分析框已按用户反馈移除——此断言失败先查包装函数）
        assert all(isinstance(t, tuple) and len(t) == 3 for t in outputs), \
            "每个 yield 必须是 (统计行, 汇总文本, 按钮状态) 三元组"
        assert "处理总量 3" in outputs[0][0], "统计行必须显示处理总量"
        # 输出文件：跟随输入格式 + 同名 txt
        assert (out / "p1.jpg").exists() and (out / "p1.txt").exists()
        assert (out / "p2.png").exists() and (out / "p2.txt").exists()
        assert (out / "p1.txt").read_text(encoding="utf-8") == "分析文本 for test"
        # 跳过逻辑：已存在的 skipme.bmp 不重写
        assert (out / "skipme.bmp").exists() and not (out / "skipme.txt").exists()
        # 汇总文件与计数
        body = (out / "batch_summary.txt").read_text(encoding="utf-8")
        assert "成功 2" in body and "跳过 1" in body and "失败 0" in body, body
        assert outputs[-1][1].rstrip() == body.rstrip(), "末帧汇总文本 = batch_summary.txt 全文"
        assert "已成功" in outputs[-1][0] and "2" in outputs[-1][0], "末帧统计行含成功数"
        # 按钮状态：中间禁用（interactive=False）、最后一帧恢复 True
        # （gradio 6 的 gr.update() 返回 dict）
        assert outputs[0][2].get("interactive") is False
        assert outputs[-1][2].get("interactive") is True
    print("[OK] batch_loop end-to-end (3-tuple stream / stats line / follow-ext / txt / skip / summary)")


def test_device_util():
    from device_util import adaptive_render_chunk, resolve_device_and_dtype
    device, dtype = resolve_device_and_dtype("auto")
    assert dtype == (torch.float32 if device.type == "cpu" else dtype)
    if device.type == "cpu":
        assert adaptive_render_chunk(-1, device) == 262144
        assert adaptive_render_chunk(131072, device) == 131072
        print("[OK] device_util on this machine: CPU -> chunk 262144")
    else:
        vram = torch.cuda.get_device_properties(device).total_memory / 1024**3
        expected = -1 if vram >= 12 else 262144
        assert adaptive_render_chunk(-1, device) == expected
        print(f"[OK] device_util on this machine: GPU {vram:.1f}GB -> chunk {expected}")


if __name__ == "__main__":
    test_params_roundtrip()
    test_layout_mapping()
    test_save_result_formats()
    test_denoise_and_exif()
    test_batch_helpers()
    test_batch_loop_end_to_end()
    test_device_util()
    test_build_ui()
    print("ALL SMOKE TESTS PASSED")
