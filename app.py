#
# VeraRetouch Gradio Web UI (CPU / GPU adaptive)
# Reuses the inference pipelines of inference.py and infer_ref_retouch.py.
# The main model is loaded once at startup and kept in memory.
#
# UI restructure per 界面汉化_打包安装_参数滑块_改造方案.md:
#   - full Chinese interface (ui/i18n.py)                                  [方案一 §2]
#   - param tab rewritten as 38 sliders (graphical input only; the JSON
#     expert mode was removed in user feedback round 2026-11-05)           [方案二 §3]
#   - GPU-priority device adaptation is handled by device_util.py          [方案三 §4.10]
#
import argparse
import ctypes
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
import yaml
from box import Box
from PIL import Image
from transformers import AutoTokenizer

from data.infer_dataset import (
    DataCollatorForUnifiedTestDataset,
    Infer_Auto_Dataset,
    Infer_Param_Dataset,
    Infer_Style_Dataset,
)
from device_util import adaptive_render_chunk, device_banner, resolve_device_and_dtype
from infer_ref_retouch import input_img_preprocess, ref_img_preprocess
from llava.constants import (
    DEFAULT_RETOUCH_COLORMIXER_TOKEN,
    DEFAULT_RETOUCH_COLORTEMP_TOKEN,
    DEFAULT_RETOUCH_LIGHT_TOKEN,
)
from llava.model.VeraRetouch import VeraRetouchForCausalLLM_Unified
from llava.utils import disable_torch_init
from ui import i18n
from ui.params_def import (
    PARAM_DEFS,
    SLIDER_KEYS,
    json_to_sliders,
    layout_order_keys,
    sliders_to_json,
    values_from_param_file,
)
from utils import get_model as get_retouch_renderer
from utils import tensor2img


class TokenProgressStreamer:
    """Minimal generate() streamer that advances a gradio progress bar per token."""

    def __init__(self, progress, max_tokens):
        self.progress = progress
        self.max_tokens = max_tokens
        self.count = 0

    def put(self, value):
        try:
            self.count += int(value.shape[-1]) if hasattr(value, "shape") else 1
            if self.progress is not None:
                frac = 0.25 + 0.65 * min(1.0, self.count / max(1, self.max_tokens))
                self.progress(frac, desc=i18n.PROGRESS_TEXTGEN_FMT.format(count=self.count))
        except Exception:
            pass

    def end(self):
        pass


class VeraRetouchRunner:
    """Loads the VeraRetouch VLM once and serves auto/style/param retouching."""

    def __init__(self, args):
        self.args = args
        self.device, self.dtype = resolve_device_and_dtype(args.dtype)
        if self.device.type == "cpu":
            # Leave one thread for system responsiveness
            torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
        self.model = None
        self.tokenizer = None

    def load(self):
        if self.model is not None:
            return
        model_path = os.path.expanduser(self.args.model_path)
        print(f"[VeraRetouch] loading model from {model_path} on {self.device} ({self.dtype}) ...")
        t0 = time.time()

        with open(self.args.config_add_path, "r", encoding="utf-8") as f:
            config_add = Box(yaml.safe_load(f))
        config_add.project_name = "test_no_instruct"
        config_add.freeze_retouch_decoder = True

        disable_torch_init()
        # model_max_length（输入侧编码截断上限）与生成参数 --max_new_tokens 是两个量纲，
        # 不可混用：默认不传该 kwarg，让 tokenizer_config.json 中训练方保存的
        # model_max_length（8192）生效；需要显式覆盖时用新增的 --model_max_length 参数。
        # （此前误传 self.args.max_new_tokens=4096，会把超过 4096 的输入序列截短。2026-11-05 修复）
        tokenizer_kwargs = dict(padding_side="right", use_fast=False)
        if self.args.model_max_length is not None:
            tokenizer_kwargs["model_max_length"] = self.args.model_max_length
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir="./cache",
            **tokenizer_kwargs,
        )
        model = VeraRetouchForCausalLLM_Unified.from_pretrained(
            model_path,
            config_add=config_add,
            cache_dir="./cache",
            torch_dtype=self.dtype,
        ).to(self.device)

        retouch_token_light_idx = tokenizer(DEFAULT_RETOUCH_LIGHT_TOKEN, add_special_tokens=False).input_ids[0]
        retouch_token_colortemp_idx = tokenizer(DEFAULT_RETOUCH_COLORTEMP_TOKEN, add_special_tokens=False).input_ids[0]
        retouch_token_colormixer_idx = tokenizer(DEFAULT_RETOUCH_COLORMIXER_TOKEN, add_special_tokens=False).input_ids[0]
        model.register_special_token_idx(
            retouch_token_light_idx, retouch_token_colortemp_idx, retouch_token_colormixer_idx
        )
        model.generation_config.pad_token_id = tokenizer.pad_token_id

        self.model = model
        self.tokenizer = tokenizer
        print(f"[VeraRetouch] model loaded in {time.time() - t0:.1f}s")

    def run(self, mode, img_path, prompt=None, param_json=None, work_dir=None, progress=None):
        """Run one retouch job. Returns (PIL output image, model text)."""
        self.load()
        if progress is not None:
            progress(0.1, desc=i18n.PROGRESS_PREPROCESS)

        image_processor = self.model.get_vision_tower().image_processor
        if mode == "auto":
            dataset = Infer_Auto_Dataset(
                img_paths=[img_path], tokenizer=self.tokenizer, image_processor=image_processor
            )
        elif mode == "style":
            dataset = Infer_Style_Dataset(
                img_paths=[img_path], prompts=[prompt], tokenizer=self.tokenizer, image_processor=image_processor
            )
        elif mode == "param":
            # Infer_Param_Dataset reads the operator parameters from a JSON file
            param_path = os.path.join(work_dir, "param.json")
            with open(param_path, "w", encoding="utf-8") as f:
                json.dump(json.loads(param_json), f, ensure_ascii=False)
            dataset = Infer_Param_Dataset(
                img_paths=[img_path], instruction_paths=[param_path],
                tokenizer=self.tokenizer, image_processor=image_processor,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        collator = DataCollatorForUnifiedTestDataset(tokenizer=self.tokenizer)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collator
        )
        batch = next(iter(loader))

        input_ids = batch["input_ids"].to(self.device)
        images = batch.get("images", None)
        if images is not None:
            images = [t.to(device=self.device, dtype=self.dtype) for t in images]
        attention_mask = batch["attention_mask"].to(self.device)
        input_imgs = [t.to(device=self.device, dtype=self.dtype) for t in batch["input_imgs"]]
        retouch_masks = [t.to(device=self.device, dtype=self.dtype) for t in batch["retouch_masks"]]

        if progress is not None:
            progress(0.25, desc=i18n.PROGRESS_TEXTGEN_START)
        streamer = TokenProgressStreamer(progress, self.args.max_new_tokens)
        try:
            with torch.inference_mode():
                retouched_imgs, decoded_outputs = self.model._generate(
                    tokenizer=self.tokenizer,
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    images=images,
                    image_sizes=batch["image_sizes"],
                    retouch_masks=retouch_masks,
                    input_imgs=input_imgs,
                    do_sample=True if self.args.temperature > 0 else False,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    num_beams=self.args.num_beams,
                    max_new_tokens=self.args.max_new_tokens,
                    use_cache=True,  # config.json ships use_cache=false; KV cache is vital on CPU
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    chunk=self.args.chunk,
                    streamer=streamer,
                )
        except Exception as e:
            raise RuntimeError(i18n.ERR_GENERATION_FMT.format(error=e)) from e

        if progress is not None:
            progress(0.95, desc=i18n.PROGRESS_RENDER)
        # retouched_imgs[0] is a BGR numpy array (see VeraRetouch._generate)
        out_rgb = cv2.cvtColor(retouched_imgs[0], cv2.COLOR_BGR2RGB)
        return Image.fromarray(out_rgb), decoded_outputs[0]


class ReferenceRetouchRunner:
    """Encoder-Renderer reference-based retouching (small standalone model, lazy-loaded)."""

    def __init__(self, args):
        self.args = args
        self.device, self.dtype = resolve_device_and_dtype(args.dtype)
        self.model = None

    def load(self):
        if self.model is not None:
            return
        print(f"[EncoderRenderer] loading {self.args.pretrained_path} on {self.device} ...")
        model = get_retouch_renderer(self.args.model_name)
        model.load_state_dict(
            torch.load(self.args.pretrained_path, map_location=self.device)["model_state_dict"]
        )
        model.to(self.device)
        model.eval()
        self.model = model
        print("[EncoderRenderer] model loaded")

    def run(self, input_img_path, ref_before_path, ref_after_path, progress=None):
        self.load()
        if progress is not None:
            progress(0.3, desc=i18n.PROGRESS_REF)
        with torch.no_grad():
            input_img = input_img_preprocess(input_img_path).to(self.device)
            ref_input_img = ref_img_preprocess(ref_before_path).to(self.device)
            ref_target_img = ref_img_preprocess(ref_after_path).to(self.device)
            mask = torch.tensor([1.0, 1.0, 1.0]).unsqueeze(0).to(self.device)
            img_adj_pred = self.model(input_img, ref_input_img, ref_target_img, mask, self.args.chunk)
            img_adj_pred = img_adj_pred.clamp(0, 1)
            return Image.fromarray(tensor2img(img_adj_pred), mode="RGB")


def make_work_dir(root, mode):
    out_dir = os.path.join(root, f"{mode}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# 输出降噪档位 → 细节回贴比例 α（final = dn + α×(orig−dn)）。
# α 大=贴回的原图细节/残余噪声多（细节优先）；α 小=更干净。默认 off=不降噪。
DENOISE_ALPHAS = {"light": 0.6, "medium": 0.35, "strong": 0.15}


def denoise_output(out_pil, alpha=0.35, h=10, template_window_size=7, search_window_size=21):
    """输出降噪：细节回贴式（方案一，2026-11-06）。

    2026-11-05 画质排查结论（tests/_quality_probe*.py）：管线无损，暗部噪点是
    "模型提亮斜率×输入传感器噪声"的物理放大（Y 高频 5.9 级/色度 0.5 级）。
    做法：亮度 Y 用 NLM(h=10) 强降噪 → 再把「原图−降噪图」差值（高频细节层）
    按比例 α 贴回：final = Ydn + α×(Yorig − Ydn)；色度 Cr/Cb 原样不动。
    用户实测裸 NLM(h=10) 细节损失过大，此回贴式在去噪与细节间可调平衡：
    α=0.6（轻）噪点约剩两成、织物/树针纹理基本可见；真机对比图见
    gradio_outputs/denoise_cmp*。全 12MP 约 2.3s。
    """
    arr = np.asarray(out_pil.convert("RGB"))[:, :, ::-1]        # RGB → BGR
    ycc = cv2.cvtColor(arr, cv2.COLOR_BGR2YCrCb)
    y_orig = ycc[..., 0].astype(np.float32)
    y_dn = cv2.fastNlMeansDenoising(y_orig.astype(np.uint8), None, h,
                                    template_window_size, search_window_size)
    restored = y_dn.astype(np.float32) + alpha * (y_orig - y_dn.astype(np.float32))
    ycc[..., 0] = np.clip(restored + 0.5, 0, 255).astype(np.uint8)  # 事后再四舍五入保持中性
    merged = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
    return Image.fromarray(merged[:, :, ::-1])                  # BGR → RGB


# EXIF 方向 -> PIL 转置操作（与 PIL ImageOps.exif_transpose 的映射一致）。
# 背景：PIL 保存时不带 EXIF，方向标签会丢——输出在相册里横竖颠倒（实测
# F 盘 11 张里有 1 张方向=8 的照片中招）。保存前按源图方向"转正"像素。
EXIF_ORIENTATION_OPS = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}


def apply_denoise(out_pil, level):
    """按页签「输出降噪」档位值处理：off=原样返回，light/medium/strong→细节回贴降噪。"""
    if level in (None, "", "off"):
        return out_pil
    return denoise_output(out_pil, alpha=DENOISE_ALPHAS.get(level, 0.35))


def bake_exif_orientation(out_pil, src_path):
    """按源图 EXIF 方向把输出像素转正（输出不再依赖方向标签，任何看图软件一致）。"""
    if not src_path:
        return out_pil
    try:
        with Image.open(src_path) as src_im:   # 只读头部 EXIF，不解码整图
            ori = (src_im.getexif() or {}).get(274)
        op = EXIF_ORIENTATION_OPS.get(ori)
        if op is not None:
            out_pil = out_pil.transpose(op)
    except Exception:
        pass  # EXIF 读不到（如自定义格式）就原样输出
    return out_pil


def save_result(out_pil, text, work_dir, fmt_key="auto", src_path=None):
    """归档修图结果，返回实际保存的图片文件路径。

    fmt_key: "auto" 跟随输入图格式 / "png" / "jpg" / "webp"；JPG/WebP 质量固定 95。
    UI 处理器返回这个路径（而不是 PIL 对象）：Gradio 的 Image 组件收到文件路径时
    原样展示、不再重编码；传 PIL 对象会被前端默认编码成 webp——这正是
    "修好的图保存格式是 webp" 的根因（gradio 6 的 gr.Image 默认 format="webp"）。
    """
    ext_map = {"png": ".png", "jpg": ".jpg", "webp": ".webp"}
    # EXIF 方向转正：PIL 保存不写 EXIF，方向标签会丢（2026-11-05 实测 11 张里
    # 1 张方向=8 的照片输出后横竖颠倒）。保存前按源图方向转正像素、不再携带标签。
    out_pil = bake_exif_orientation(out_pil, src_path)
    pil_fmt_map = {  # PIL 的格式名与扩展名不同名（.jpg -> JPEG）
        ".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".webp": "WEBP",
        ".bmp": "BMP", ".tif": "TIFF", ".tiff": "TIFF",
    }
    if fmt_key == "auto":
        src_ext = os.path.splitext(src_path or "")[1].lower()
        if src_ext in (".jpg", ".jpeg"):
            src_ext = ".jpg"
        ext = src_ext if src_ext in (".png", ".jpg", ".webp", ".bmp", ".tif", ".tiff") else ".png"
    else:
        ext = ext_map[fmt_key]
    img_path = os.path.join(work_dir, f"retouched{ext}")
    save_kwargs = {}
    pil, fmt = out_pil, pil_fmt_map[ext]
    if fmt == "JPEG":  # .jpg / .jpeg：PIL 的 JPG 不支持 alpha 通道
        save_kwargs["quality"] = 95
        if pil.mode not in ("RGB", "L"):
            pil = pil.convert("RGB")
    elif fmt == "WEBP":
        save_kwargs["quality"] = 95
    pil.save(img_path, format=fmt, **save_kwargs)
    if text:
        with open(os.path.join(work_dir, "analysis.txt"), "w", encoding="utf-8") as f:
            f.write(text)
    return img_path


# 说明（2026-11-06）：相机 RAW 支持（rawpy 解码、26 种扩展名、gr.Image 拦截、
# 前端 accepted_types 补丁）曾按用户需求实现并上线，后按用户决定整体回退——
# 目前不支持相机 RAW（CR2/CR3/NEF/ARW/DNG 等），仅支持常规图片格式。
# 保留的 _patch_gradio_asset_no_cache()（下方）与本次回退无关：它修复的是
# gradio 资产响应缺缓存控制头导致的前端更新不生效问题，对任何界面改动都有益。
# 相关排查/样张留档：tests/_check_rawpy.py、tests/_sample_raw/。


def _patch_gradio_asset_no_cache():
    """gradio 的 /assets/* 响应不带 Cache-Control 头 → 浏览器按"文件年龄 10%"
    启发式缓存数天（gradio 轮子里的 JS 是几个月前构建的）。后果：改过前端
    编译资产（或 gradio 自身升级换名）后，用户浏览器迟迟拿不到新 JS。
    包一层 FileResponse 强制 no-cache：浏览器每次都带 If-None-Match 回源
    校验，内容没变时服务器 304、几乎零开销；变了立刻拿到新 JS。"""
    from gradio import routes as _g_routes
    from starlette.responses import FileResponse as _StarletteFileResponse
    from starlette.responses import Response as _PlainResponse

    _orig = _g_routes.FileResponse
    if getattr(_orig, "_vr_no_cache", False):
        return  # 已补丁过（重复 import app 时）

    class _NoCacheFileResponse(_StarletteFileResponse):
        _vr_no_cache = True

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.headers.setdefault("Cache-Control", "no-cache")

        async def __call__(self, scope, receive, send):
            # 该 Starlette 版本（1.10）的 FileResponse 在 __init__ 时不设 stat 头，
            # etag 要等 __call__ 里 os.stat 才写入——所以这里先补设再比对，
            # 且父类看到 stat_result 非 None 会跳过重复设置（行为不变）。
            if self.stat_result is None:
                try:
                    self.set_stat_headers(os.stat(self.path))
                except FileNotFoundError:
                    await super().__call__(scope, receive, send)
                    return
            etag = self.headers.get("etag")
            if etag:
                for key, val in scope.get("headers", []):
                    if key == b"if-none-match" and val.decode("latin-1").strip() == etag:
                        await _PlainResponse(status_code=304)(scope, receive, send)
                        return
            await super().__call__(scope, receive, send)

    _g_routes.FileResponse = _NoCacheFileResponse


_patch_gradio_asset_no_cache()


# ===================== 批量自动修图（技术方案 §4.5–§4.9，2026-11-05） =====================

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
GC_EVERY = 10  # 每 10 张做一次 gc/显存缓存清理保险丝（§4.9：低频定时收 + 出事立刻收）
BATCH_DEFAULT_OUT_DIR_NAME = "veraretouch_批修输出"


def scan_images(in_dir):
    """扫描顶层可处理图片：扩展名过滤（大小写不敏感）、隐藏文件排除、按文件名排序。

    in_dir 不存在/不可读时返回 None；存在时返回 pathlib.Path 列表（可为空列表）。
    不递归子目录（技术方案 §7-D4：只处理顶层）。
    """
    d = Path(in_dir) if in_dir else None
    if d is None or not d.is_dir():
        return None
    return sorted(
        p for p in d.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMG_EXTS
    )


def _native_pick_dir(title):
    """弹出 Windows 原生"选择文件夹"对话框，返回所选路径；取消/失败返回 None。

    走 PowerShell FolderBrowserDialog：嵌入式 runtime 没带 tkinter（实测），
    而 powershell.exe 在所有 Windows 上可用，零第三方依赖。输出强制 UTF-8，
    避免中文路径被 GBK 控制台编码打碎。
    """
    script = (
        "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8\n"
        "Add-Type -AssemblyName System.Windows.Forms\n"
        "$d = New-Object System.Windows.Forms.FolderBrowserDialog\n"
        f"$d.Description = '{title}'\n"
        "$r = $d.ShowDialog()\n"
        "if ($r -eq [System.Windows.Forms.DialogResult]::OK) { Write-Output $d.SelectedPath }\n"
    )
    tmp_ps = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_pick_dir_tmp.ps1")
    try:
        with open(tmp_ps, "w", encoding="utf-8-sig") as f:  # PowerShell 5 要求 UTF-8 BOM
            f.write(script)
        out = subprocess.run(
            ["powershell", "-NoProfile", "-STA", "-ExecutionPolicy", "Bypass", "-File", tmp_ps],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
        )
        path = (out.stdout or "").strip().splitlines()[-1] if (out.stdout or "").strip() else ""
        return path if path and os.path.isdir(path) else None
    except Exception:
        return None
    finally:
        try:
            os.remove(tmp_ps)
        except OSError:
            pass


def overall_progress(i, n, durations, cur_elapsed):
    """总体进度条的时间加权进度与 ETA（技术方案 §4.5，实现 (3)）。

    第 1 张（无历史耗时）退化为线性 (i-1)/n；此后
        prog = (已完成累计耗时 + 正在进行这张按均值折算的部分) / (n × 平均单张耗时)
    折算比例上限 0.95、总体上限 0.99——最后一步只认真实完成，绝不虚标。
    返回 (0..1 浮点、ETA 秒数或 None)。
    """
    if not durations:
        return max((i - 1) / n, 0.0), None
    avg = sum(durations) / len(durations)
    credit = min(cur_elapsed / avg, 0.95) if avg > 0 else 0.0
    prog = (sum(durations) + credit * avg) / (n * avg)
    return max(min(prog, 0.99), 0.0), avg * (n - (i - 1))


def _fmt_eta(seconds):
    if seconds is None:
        return "统计中…"
    seconds = int(seconds)
    return f"{seconds // 60}分{seconds % 60:02d}秒"


def _win_beep():
    """Windows 系统提示音（零依赖；非 Windows 或静音时自动无效）。"""
    try:
        if os.name == "nt":
            ctypes.windll.user32.MessageBeep(0)  # MB_OK 默认"叮"
    except Exception:
        pass


def _default_out_dir(in_dir):
    d = Path(in_dir)
    return str(d / BATCH_DEFAULT_OUT_DIR_NAME)


def _already_done(img_path, out_dir):
    """断点续跑判断：输出目录已存在同基名、同格式（或 png 回退）的结果图。"""
    base = Path(img_path).stem
    return (Path(out_dir) / f"{base}{Path(img_path).suffix.lower()}" ).exists()


def _write_summary(out_dir, begin_time, end_time, ok, fail, skip, lines, in_dir):
    """写入 batch_summary.txt 并返回全文文本（同时供界面"完成汇总"展示）。"""
    total = ok + fail + skip
    elapsed = (end_time - begin_time) if end_time >= begin_time else 0.0
    avg = elapsed / total if total else 0.0
    body = "\n".join(
        [
            "==== VeraRetouch 批量修图汇总 ====",
            f"开始时间  : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(begin_time))}",
            f"结束时间  : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}",
            f"输入目录  : {in_dir}",
            f"输出目录  : {out_dir}",
            f"总计     : {total} 张    成功 {ok}    失败 {fail}    跳过 {skip}",
            f"总耗时    : {int(elapsed // 60)} 分 {int(elapsed % 60):02d} 秒    平均 {avg:.1f} 秒/张",
            "-" * 72,
            *lines,
        ]
    )
    summary_path = Path(out_dir) / "batch_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(body + "\n")
    return body


def _fmt_batch_stats(total, ok, fail, skip, cur_name=None):
    """汇总滚动框上方的实时统计行（此前是"预览图+分析框"，按用户 2026-11-05 反馈精简）。"""
    line = i18n.BATCH_STATS_FMT.format(total=total, ok=ok, fail=fail, skip=skip)
    if cur_name:
        line += i18n.BATCH_STATS_CUR_FMT.format(name=cur_name)
    return line


def batch_loop(runner, in_dir, out_dir, overall_prog=gr.Progress(), per_img_prog=gr.Progress(),
               denoise_level="off"):
    """批量自动修图主循环（generator）：逐张 auto 模式 → 保存同格式图 + 同名分析 txt。

    denoise_level: 输出降噪档位（off/light/medium/strong，默认 off=模型裸输出；
    light/medium/strong 走细节回贴降噪，见 apply_denoise/denoise_output）。

    防溢出纪律（技术方案 §4.9）：循环内绝不累积图片对象；每张完成即 del 引用；
    每 GC_EVERY 张 gc.collect()（GPU 机顺带 empty_cache()）；显存 OOM 先清缓存
    重试一次本图，仍失败仅记单张失败、不伤整批。
    """
    in_dir = (in_dir or "").strip().strip('"')
    if not os.path.isdir(in_dir):
        raise gr.Error(i18n.BATCH_ERR_NO_INDIR.format(path=in_dir or "空白"))
    files = scan_images(in_dir)
    if not files:
        raise gr.Error(i18n.BATCH_ERR_NO_IMG)

    out_dir = (out_dir or "").strip().strip('"') or _default_out_dir(in_dir)
    if os.path.abspath(out_dir) == os.path.abspath(in_dir):
        raise gr.Error(i18n.BATCH_ERR_SAME_DIR)
    os.makedirs(out_dir, exist_ok=True)
    # 输出目录注册为静态路径：浏览器直接按原路径访问这些文件、不搬进 gradio 缓存。
    # 否则 yield 出去的 F:\…\result.jpg 会触发 InvalidPathError（不在 cwd/temp 白名单）。
    # 语义提示：该目录下所有文件此后都可通过 /file= 直接访问——本应用是本机单用户
    # 场景，且该目录本就是用户为"修图结果"选定的目录，无越权风险。
    gr.set_static_paths([out_dir])

    # 批内临时区（放输出目录下，保证 os.replace/shutil.move 同盘；上次残留自动清）
    tmp_dir = Path(out_dir) / "_batch_tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)
    tmp_dir_s = str(tmp_dir)

    n = len(files)
    btn_off = gr.update(interactive=False)
    btn_on = gr.update(interactive=True)
    summary_lines = [i18n.BATCH_LOG_HEADER]
    ok = fail = skip = 0
    durations = []
    begin_time = time.time()

    # 先发一条"已启动"事件：立即禁用开始按钮，避免批处理期间重复点击
    yield (_fmt_batch_stats(n, 0, 0, 0), "\n".join(summary_lines), btn_off)

    try:
        for i, img_path in enumerate(files, start=1):
            name = img_path.name
            stem = img_path.stem
            t0 = time.perf_counter()
            prog, eta = overall_progress(i, n, durations, 0.0)
            overall_prog(
                prog,
                desc=i18n.BATCH_OVERALL_FMT.format(i=i, n=n, name=name, eta=_fmt_eta(eta)),
            )
            # 断点续跑：目标输出已存在 → 跳过（不重复耗算力）
            if _already_done(img_path, out_dir):
                skip += 1
                summary_lines.append(f"{name}  →  ✓ 跳过（输出已存在）")
                yield (_fmt_batch_stats(n, ok, fail, skip), "\n".join(summary_lines), btn_off)
                continue
            try:
                out_pil, text = runner.run("auto", str(img_path), progress=per_img_prog)
            # CUDA OOM：清显存缓存后重试一次本图（§4.9.2 第 2 层）
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                try:
                    out_pil, text = runner.run("auto", str(img_path), progress=per_img_prog)
                except Exception as e:
                    fail += 1
                    durations.append(time.perf_counter() - t0)
                    summary_lines.append(f"{name}  →  ✗ 失败(显存不足重试无效): {e}")
                    yield (_fmt_batch_stats(n, ok, fail, skip), "\n".join(summary_lines), btn_off)
                    continue
            except Exception as e:
                fail += 1
                durations.append(time.perf_counter() - t0)
                summary_lines.append(f"{name}  →  ✗ 失败: {e}")
                gr.Warning(i18n.BATCH_IMG_FAILED_WARN.format(name=name))
                yield (_fmt_batch_stats(n, ok, fail, skip), "\n".join(summary_lines), btn_off)
                continue

            # 保存：先落临时区再原子改名到最终名（跟随输入格式 + 同名分析 txt）
            out_pil = apply_denoise(out_pil, denoise_level)
            tmp_img = save_result(out_pil, None, tmp_dir_s, fmt_key="auto", src_path=str(img_path))
            ext = os.path.splitext(tmp_img)[1].lower()
            final_img = str(Path(out_dir) / f"{stem}{ext}")
            shutil.move(tmp_img, final_img)
            analysis_txt = Path(out_dir) / f"{stem}.txt"
            analysis_txt.write_text(text or "", encoding="utf-8")
            ok += 1
            spent = time.perf_counter() - t0
            durations.append(spent)
            summary_lines.append(
                f"{name}  →  {stem}{ext}  {_fmt_eta(spent)}  分析记录: {stem}.txt"
            )
            yield (_fmt_batch_stats(n, ok, fail, skip), "\n".join(summary_lines), btn_off)

            # 内存保险丝：低频清理（§4.9）
            if i % GC_EVERY == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del out_pil
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    end_time = time.time()
    body = _write_summary(out_dir, begin_time, end_time, ok, fail, skip, summary_lines[1:], in_dir)
    _win_beep()
    gr.Info(i18n.BATCH_DONE.format(ok=ok, fail=fail, skip=skip, dir=out_dir))
    yield (_fmt_batch_stats(n, ok, fail, skip), body, btn_on)


def build_ui(runner, ref_runner, args):
    device_desc = device_banner(runner.device, runner.dtype)
    sample_param_path = os.path.join("data_samples", "param.json")

    def _run_retouch(mode, image, prompt, param_text, progress, export_fmt="auto", denoise_level="off"):
        if image is None:
            raise gr.Error(i18n.ERR_NEED_IMAGE)
        if mode == "style" and not (prompt or "").strip():
            raise gr.Error(i18n.ERR_NEED_STYLE)
        if mode == "param":
            # 输入一律来自滑块（滑块→JSON 内部转换），参数文本恒合法；仅做全 0 提示
            values = json_to_sliders(param_text)[0]
            if all(v == 0 for v in values):
                gr.Warning(i18n.WARN_ALL_ZERO_SUBMIT)  # non-blocking per §3.4 P5
        work_dir = make_work_dir(args.output_dir, mode)
        out_pil, text = runner.run(mode, image, prompt=prompt, param_json=param_text,
                                   work_dir=work_dir, progress=progress)
        out_pil = apply_denoise(out_pil, denoise_level)
        progress(1.0, desc=i18n.PROGRESS_DONE)
        img_path = save_result(out_pil, text, work_dir, fmt_key=export_fmt, src_path=image)
        # 返回文件路径而非 PIL：见 save_result 文档（避免前端 webp 重编码）
        return img_path, text or "(no text output)"

    # One handler per mode: gradio maps click() inputs to handler args by position,
    # so each mode must receive its own argument list (see CPU_MIGRATION plan note).
    def auto_handler(image, export_fmt, denoise_level, progress=gr.Progress()):
        return _run_retouch("auto", image, None, None, progress, export_fmt, denoise_level)

    def style_handler(image, prompt, export_fmt, denoise_level, progress=gr.Progress()):
        return _run_retouch("style", image, prompt, None, progress, export_fmt, denoise_level)

    def param_handler(image, export_fmt, denoise_level, *slider_values, progress=gr.Progress()):
        """Sliders-only input: the 38 layout-ordered sliders drive the pipeline.
        (JSON 专家模式按用户反馈于 2026-11-05 移除；滑块→JSON 转换仅供内部管线使用。)"""
        param_text = sliders_to_json(*_layout_to_protocol(slider_values))
        return _run_retouch("param", image, None, param_text, progress, export_fmt, denoise_level)

    def ref_handler(input_img, ref_before, ref_after, export_fmt, denoise_level,
                    progress=gr.Progress()):
        # 参考图必须齐全（用户要求 2026-11-05）：
        # 原实现的"修图前留空则用输入图顶替"的模式会让无参考图也照跑、产出毫无
        # 意义的结果；现改为缺哪张就弹窗报错、不允许执行。gr.Error 在 Gradio
        # 中以弹窗（模态提示）呈现，不会被静默吞掉。
        if input_img is None:
            raise gr.Error(i18n.ERR_REF_NEED_INPUT)
        if ref_before is None:
            raise gr.Error(i18n.ERR_REF_NEED_BEFORE)
        if ref_after is None:
            raise gr.Error(i18n.ERR_REF_NEED_AFTER)
        out_pil = ref_runner.run(input_img, ref_before, ref_after, progress=progress)
        out_pil = apply_denoise(out_pil, denoise_level)
        progress(1.0, desc=i18n.PROGRESS_DONE)
        work_dir = make_work_dir(args.output_dir, "reference")
        img_path = save_result(out_pil, None, work_dir, fmt_key=export_fmt, src_path=input_img)
        return img_path

    def _reset_sliders():
        return [0] * len(slider_layout_keys)

    def batch_handler(in_dir, out_dir, denoise_level, overall_prog=gr.Progress(), per_img_prog=gr.Progress()):
        """批量主页任务薄包装：把 runner 闭包传给模块层 batch_loop。

        注意必须是生成器委托（yield from 而非 return）：gradio 校验 handler
        返回值数量，return 一个 generator 对象会被当成"只返回了 1 个输出"而报
        ValueError（本次实测踩坑）；yield from 让本函数自身成为生成器。
        gr.Progress 形参保留在本函数签名上（两条进度条在事件语境里按位绑定，
        见技术方案 §4.5 与其风险注记；调用 batch_loop 时把同一批对象传下去，
        内部更新仍作用于同一个条）。
        """
        yield from batch_loop(runner, in_dir, out_dir, overall_prog, per_img_prog,
                              denoise_level=denoise_level)
    def _load_sample():
        try:
            return _protocol_to_layout(values_from_param_file(sample_param_path))
        except FileNotFoundError:
            raise gr.Error("未找到 data_samples/param.json（示例参数文件缺失）")

    with gr.Blocks(title=i18n.TITLE) as demo:
        device_note = f"> {i18n.DEVICE_NOTE.format(device=device_desc)}"
        if runner.device.type == "cpu":
            device_note += i18n.CPU_NOTE
        gr.Markdown(f"# {i18n.PAGE_HEADER}\n{i18n.PAGE_SUBHEADER}\n\n{device_note}")

        with gr.Tab(i18n.TAB_AUTO):
            with gr.Row():
                with gr.Column():
                    auto_input = gr.Image(label=f"{i18n.INPUT_IMAGE_LABEL}（{i18n.INPUT_IMAGE_INFO}）",
                                          type="filepath")
                    auto_fmt = gr.Dropdown(choices=i18n.EXPORT_FORMAT_CHOICES, value="auto",
                                           label=i18n.EXPORT_FORMAT_LABEL, info=i18n.EXPORT_FORMAT_INFO)
                    auto_dn = gr.Dropdown(choices=i18n.DENOISE_CHOICES, value="off",
                                          label=i18n.DENOISE_LABEL, info=i18n.DENOISE_INFO)
                    auto_btn = gr.Button(i18n.BTN_START_AUTO, variant="primary")
                auto_output = gr.Image(label=i18n.OUTPUT_IMAGE_LABEL, interactive=False)
            auto_text = gr.Textbox(label=i18n.ANALYSIS_LABEL, lines=12, interactive=False)
            auto_btn.click(auto_handler, inputs=[auto_input, auto_fmt, auto_dn],
                           outputs=[auto_output, auto_text])

        with gr.Tab(i18n.TAB_STYLE):
            style_prompt = gr.Textbox(
                label=i18n.STYLE_PROMPT_LABEL, placeholder=i18n.STYLE_PROMPT_PLACEHOLDER,
                info=i18n.STYLE_PROMPT_INFO, lines=1,
            )
            with gr.Row():
                with gr.Column():
                    style_input = gr.Image(label=f"{i18n.INPUT_IMAGE_LABEL}（{i18n.INPUT_IMAGE_INFO}）",
                                           type="filepath")
                    style_fmt = gr.Dropdown(choices=i18n.EXPORT_FORMAT_CHOICES, value="auto",
                                            label=i18n.EXPORT_FORMAT_LABEL, info=i18n.EXPORT_FORMAT_INFO)
                    style_dn = gr.Dropdown(choices=i18n.DENOISE_CHOICES, value="off",
                                           label=i18n.DENOISE_LABEL, info=i18n.DENOISE_INFO)
                    style_btn = gr.Button(i18n.BTN_START_STYLE, variant="primary")
                style_output = gr.Image(label=i18n.OUTPUT_IMAGE_LABEL, interactive=False)
            style_text = gr.Textbox(label=i18n.ANALYSIS_LABEL, lines=12, interactive=False)
            style_btn.click(
                style_handler, inputs=[style_input, style_prompt, style_fmt, style_dn],
                outputs=[style_output, style_text])

        with gr.Tab(i18n.TAB_PARAM):
            gr.Markdown(f"> {i18n.PARAM_SLIDER_RANGE_INFO}")
            # Slider creation order (two-column layout) != protocol key order,
            # so we keep an explicit layout -> protocol authority pair.
            sliders = []
            slider_layout_keys = list(layout_order_keys())  # order of `sliders` list

            def _emit_slider(key):
                param = PARAM_DEFS[key]
                sliders.append(gr.Slider(
                    minimum=-100, maximum=100, step=1, value=0,
                    label=f"{param['name']}（{key}）", info=param["desc"],
                ))

            def _layout_to_protocol(vals):
                by_key = dict(zip(slider_layout_keys, vals))
                return [by_key[k] for k in SLIDER_KEYS]

            def _protocol_to_layout(vals):
                by_key = dict(zip(SLIDER_KEYS, vals))
                return [by_key[k] for k in slider_layout_keys]

            # 滑块面板（唯一输入方式；JSON 专家模式按用户反馈移除）
            with gr.Group():
                for group_id in (1, 2, 3):
                    group_keys = [k for k in SLIDER_KEYS if PARAM_DEFS[k]["group"] == group_id]
                    with gr.Accordion(
                        i18n.PARAM_GROUP_TITLES[group_id],
                        open=(group_id == 1),  # 只有第一组默认展开
                    ):
                        with gr.Row():
                            with gr.Column():
                                for key in group_keys[0::2]:
                                    _emit_slider(key)
                            with gr.Column():
                                for key in group_keys[1::2]:
                                    _emit_slider(key)
                with gr.Row():
                    btn_reset = gr.Button(i18n.BTN_RESET_ALL, variant="secondary", size="sm")
                    btn_sample = gr.Button(i18n.BTN_LOAD_SAMPLE, variant="secondary", size="sm")
            # 防错位断言：滑块组件顺序必须与布局键序一一对应（历史踩坑点，见方案 §3.4 P2）
            assert len(sliders) == len(slider_layout_keys)
            for comp, key in zip(sliders, slider_layout_keys):
                assert f"（{key}）" in comp.label, f"slider order mismatch at {key}"
            with gr.Row():
                with gr.Column():
                    param_input = gr.Image(label=f"{i18n.INPUT_IMAGE_LABEL}（{i18n.INPUT_IMAGE_INFO}）",
                                           type="filepath")
                    param_fmt = gr.Dropdown(choices=i18n.EXPORT_FORMAT_CHOICES, value="auto",
                                            label=i18n.EXPORT_FORMAT_LABEL, info=i18n.EXPORT_FORMAT_INFO)
                    param_dn = gr.Dropdown(choices=i18n.DENOISE_CHOICES, value="off",
                                           label=i18n.DENOISE_LABEL, info=i18n.DENOISE_INFO)
                    param_btn = gr.Button(i18n.BTN_START_PARAM, variant="primary")
                param_output = gr.Image(label=i18n.OUTPUT_IMAGE_LABEL, interactive=False)
            param_text = gr.Textbox(label=i18n.ANALYSIS_LABEL, lines=12, interactive=False)

            param_btn.click(
                param_handler,
                inputs=[param_input, param_fmt, param_dn] + sliders,
                outputs=[param_output, param_text],
            )
            btn_reset.click(_reset_sliders, inputs=None, outputs=sliders)
            btn_sample.click(_load_sample, inputs=None, outputs=sliders)

        with gr.Tab(i18n.TAB_REF):
            gr.Markdown(i18n.REF_INFO_MD)
            with gr.Row():
                ref_input = gr.Image(label=i18n.REF_INPUT_LABEL, type="filepath")
                ref_before = gr.Image(label=i18n.REF_BEFORE_LABEL, type="filepath")
                ref_after = gr.Image(label=i18n.REF_AFTER_LABEL, type="filepath")
            ref_fmt = gr.Dropdown(choices=i18n.EXPORT_FORMAT_CHOICES, value="auto",
                                  label=i18n.EXPORT_FORMAT_LABEL, info=i18n.EXPORT_FORMAT_INFO)
            ref_dn = gr.Dropdown(choices=i18n.DENOISE_CHOICES, value="off",
                                 label=i18n.DENOISE_LABEL, info=i18n.DENOISE_INFO)
            ref_btn = gr.Button(i18n.BTN_START_REF, variant="primary")
            ref_output = gr.Image(label=i18n.OUTPUT_IMAGE_LABEL, interactive=False)
            ref_btn.click(
                ref_handler, inputs=[ref_input, ref_before, ref_after, ref_fmt, ref_dn],
                outputs=[ref_output])

        with gr.Tab(i18n.TAB_BATCH):
            gr.Markdown(f"> {i18n.BATCH_INFO_MD}")
            # ---------- 批量页内小型交互 ----------
            def pick_in_dir(cur):
                p = _native_pick_dir(i18n.BATCH_PICK_TITLE_IN)
                return p if p else cur          # 用户取消 → 文本框保持原值

            def pick_out_dir(cur):
                p = _native_pick_dir(i18n.BATCH_PICK_TITLE_OUT)
                return p if p else cur

            def scan_handler(in_dir_text):
                d = (in_dir_text or "").strip().strip('"')
                if not os.path.isdir(d):
                    return i18n.BATCH_SCAN_MISS.format(path=d or "空白")
                files = scan_images(d)
                if not files:
                    return i18n.BATCH_SCAN_EMPTY
                counts = {"jpg": 0, "png": 0, "webp": 0, "other": 0}
                for p in files:
                    ext = p.suffix.lower()
                    if ext in (".jpg", ".jpeg"):
                        counts["jpg"] += 1
                    elif ext == ".png":
                        counts["png"] += 1
                    elif ext == ".webp":
                        counts["webp"] += 1
                    else:
                        counts["other"] += 1
                return i18n.BATCH_SCAN_FMT.format(
                    n=len(files), a=counts["jpg"], b=counts["png"],
                    c=counts["webp"], d=counts["other"],
                )

            with gr.Row():
                batch_in_dir = gr.Textbox(label=i18n.BATCH_IN_DIR_LABEL,
                                          placeholder=r"D:\photos\待修图")
                with gr.Column(min_width=110):
                    btn_browse_in = gr.Button(i18n.BTN_BROWSE, size="sm")
            with gr.Row():
                batch_out_dir = gr.Textbox(label=i18n.BATCH_OUT_DIR_LABEL,
                                           placeholder=r"D:\photos\待修图\veraretouch_批修输出")
                with gr.Column(min_width=110):
                    btn_browse_out = gr.Button(i18n.BTN_BROWSE, size="sm")
            with gr.Row():
                btn_scan = gr.Button(i18n.BTN_SCAN, variant="secondary", size="sm")
                batch_btn = gr.Button(i18n.BTN_START_BATCH, variant="primary")
            batch_dn = gr.Dropdown(choices=i18n.DENOISE_CHOICES, value="off",
                                   label=i18n.DENOISE_LABEL, info=i18n.DENOISE_INFO)
            batch_scan_info = gr.Textbox(label="扫描结果", value="尚未扫描", lines=1,
                                         interactive=False)
            # 实时统计行（总量/成功/失败/跳过）+ 汇总滚动框——按用户 2026-11-05 反馈
            # 移除"最近完成的修图结果"预览图与"最近一张分析"框，界面更清爽
            batch_stats = gr.Markdown("")
            batch_summary = gr.Textbox(label=i18n.BATCH_SUMMARY_LABEL, lines=12,
                                       interactive=False)

            btn_browse_in.click(pick_in_dir, inputs=[batch_in_dir], outputs=[batch_in_dir])
            btn_browse_out.click(pick_out_dir, inputs=[batch_out_dir], outputs=[batch_out_dir])
            btn_scan.click(scan_handler, inputs=[batch_in_dir], outputs=[batch_scan_info])
            # 批量主循环：generator 流式更新 [实时统计行, 汇总文本, 按钮状态]；
            # 两条进度条由 handler 形参里的 gr.Progress 自动绑定（技术方案 §4.5）。
            batch_btn.click(
                batch_handler,
                inputs=[batch_in_dir, batch_out_dir, batch_dn],
                outputs=[batch_stats, batch_summary, batch_btn],
            )

    return demo


def main():
    parser = argparse.ArgumentParser(description="VeraRetouch Gradio Web UI")
    parser.add_argument("--model-path", type=str, default="./checkpoints/VeraRetouch")
    parser.add_argument("--config_add_path", type=str, default="./configs/infer_config.yaml")
    parser.add_argument("--pretrained_path", type=str, default="./checkpoints/encoder_renderer.pth",
                        help="Encoder-Renderer weights for reference retouch")
    parser.add_argument("--model_name", type=str, default="RetouchRenderer_Resnet18Encoder_InputCatMixedCBAM")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float32", "bfloat16"],
                        help="compute dtype: auto = bf16 on Ampere+ GPUs, fp32 on older GPUs / CPU")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="生成参数：模型最多新生成多少个 token（推理输出长度）")
    parser.add_argument("--model_max_length", type=int, default=None,
                        help="tokenizer 输入侧编码截断上限；默认不指定，"
                             "使用 checkpoints 中 tokenizer_config.json 里训练方保存的"
                             " model_max_length（8192）。与 --max_new_tokens 相互独立")
    parser.add_argument("--chunk", type=int, default=-1,
                        help="Renderer chunk size for large images, e.g. 262144 (512*512). "
                             "-1 resolves automatically: CPU -> 262144, GPU with >= 12GB VRAM -> off.")
    parser.add_argument("--output_dir", type=str, default="./gradio_outputs")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--inbrowser", action=argparse.BooleanOptionalAction, default=True,
                        help="服务就绪后自动打开系统默认浏览器访问界面（默认开启；"
                             "用 --no-inbrowser 关闭。安装版启动器会传 --no-inbrowser，"
                             "由启动器端口探活后自行开浏览器，避免重复弹出）")
    parser.add_argument("--share", action="store_true", help="Create a public gradio share link")
    parser.add_argument("--listen", action="store_true", help="Serve on 0.0.0.0 (LAN access)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    probe_device, _ = resolve_device_and_dtype("auto")
    # Renderer memory: the pixel-wise MLP renderer needs >4 GB activations for a
    # 1707x1280 image without chunking; keep it bounded unless VRAM is plentiful.
    args.chunk = adaptive_render_chunk(args.chunk, probe_device)
    print(f"[app] renderer chunk resolved to {args.chunk} for {probe_device.type}")

    runner = VeraRetouchRunner(args)
    runner.load()  # eager load: the UI starts once the model is ready
    ref_runner = ReferenceRetouchRunner(args)  # small model, lazy-loaded on first use

    demo = build_ui(runner, ref_runner, args)
    # Single-instance model: serialize requests
    demo.queue(max_size=8)
    demo.launch(
        server_name="0.0.0.0" if args.listen else "127.0.0.1",
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser,  # 服务就绪后自动打开默认浏览器（--no-inbrowser 可关闭）
    )


if __name__ == "__main__":
    main()
