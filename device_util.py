#
# Shared device / dtype / chunk adaptation helpers.
# Used by both app.py (Gradio UI) and inference.py (CLI) so that the
# GPU-priority mode behaves identically in both entry points.
#
# Rules (see 界面汉化_打包安装_参数滑块_改造方案.md §4.10):
#   - Device: CUDA when available, otherwise CPU.
#   - dtype auto: GPU compute capability >= 8.0 (RTX 30 / Ampere+) -> bfloat16,
#                 older NVIDIA GPUs        -> float32 (still runs on GPU),
#                 CPU                      -> float32.
#   - Renderer chunk: CPU always 262144; GPU only chunks when VRAM < 12 GB
#                 (a 1707x1280 render without chunking needs ~4.5 GB of
#                 activations on top of the ~2.3 GB weights).
#
import torch

RENDER_CHUNK_CPU = 262144          # 512 * 512 pixels per renderer chunk
RENDER_CHUNK_GPU_LOW_VRAM = 262144
GPU_CHUNK_FREE_VRAM_GB = 12.0      # allow unchunked rendering above this VRAM


def resolve_device_and_dtype(dtype_arg="auto"):
    """Return (device, dtype). dtype_arg: auto | float32 | bfloat16."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype_arg != "auto" and dtype_arg is not None:
        dtype = getattr(torch, dtype_arg)
    elif device.type == "cpu":
        dtype = torch.float32
    else:
        # bf16 needs Ampere+; older NVIDIA cards fall back to fp32 (still on GPU)
        major, _ = torch.cuda.get_device_capability(device)
        dtype = torch.bfloat16 if major >= 8 else torch.float32
    return device, dtype


def adaptive_render_chunk(chunk, device):
    """Resolve chunk=-1 into a value that fits this machine's memory budget."""
    if chunk != -1:
        return chunk
    if device.type == "cpu":
        return RENDER_CHUNK_CPU
    total_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    if total_gb < GPU_CHUNK_FREE_VRAM_GB:
        return RENDER_CHUNK_GPU_LOW_VRAM
    return chunk  # big-VRAM GPU: no chunking


def device_banner(device, dtype):
    """Human-readable one-line device info for the console log / UI badge."""
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        major, _ = torch.cuda.get_device_capability(device)
        mode = "bf16" if dtype == torch.bfloat16 else "fp32 (legacy GPU, bf16 unsupported)"
        return (
            f"GPU 优先模式 | {props.name} | 显存 {props.total_memory / (1024 ** 3):.1f} GB "
            f"| 算力 {major}.x | 精度 {mode}"
        )
    return (
        f"CPU 模式 | 逻辑线程 {torch.get_num_threads()} | 精度 float32 "
        f"（单图约 2-5 分钟）"
    )
