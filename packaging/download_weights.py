#
# VeraRetouch model weights downloader (first-run setup).
# Works standalone (python download_weights.py) or via ensure_weights() from launcher.py.
# Defaults to the HF mirror per 改造方案 §4.5 / §6(C6).
#
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

APP_DIR = None  # set by ensure_weights(); CLI mode uses cwd
REPO_MAIN = "Gyh68/VeraRetouch"
REPO_ENCODER = "Gyh68/VeraRetouch.Encoder_Renderer"


def _verify(path, min_mb):
    if os.path.exists(path) and os.path.getsize(path) >= min_mb * 1024 * 1024:
        return True
    return False


def ensure_weights(app_dir=None):
    """Download missing weights. Returns True when main + encoder weights are ready."""
    app_dir = app_dir or os.getcwd()
    main_dir = os.path.join(app_dir, "checkpoints", "VeraRetouch")
    main_safetensors = os.path.join(main_dir, "model.safetensors")
    enc_path = os.path.join(app_dir, "checkpoints", "encoder_renderer.pth")

    # lazy import AFTER HF_ENDPOINT is set on a fresh interpreter
    from huggingface_hub import hf_hub_download, snapshot_download

    if not _verify(main_safetensors, 2000):
        print(f"[weights] 主模型缺失，开始从镜像下载（约 2.3 GB，请保持网络畅通）", flush=True)
        snapshot_download(
            repo_id=REPO_MAIN,
            local_dir=main_dir,
        )
        print("[weights] 主模型下载完成", flush=True)
    else:
        print("[weights] 主模型已就绪", flush=True)

    if not _verify(enc_path, 150):
        print("[weights] Encoder-Renderer 权重缺失，开始下载（约 180 MB）", flush=True)
        hf_hub_download(
            repo_id=REPO_ENCODER,
            filename="encoder_renderer.pth",
            local_dir=os.path.join(app_dir, "checkpoints"),
        )
        print("[weights] Encoder-Renderer 下载完成", flush=True)
    else:
        print("[weights] Encoder-Renderer 已就绪", flush=True)
    return True


if __name__ == "__main__":
    try:
        ensure_weights(None)
        print("[weights] 权重全部就绪")
    except KeyboardInterrupt:
        print("\n[weights] 已中断（支持断点续传，可重新运行）")
        sys.exit(1)
    except Exception as e:
        print(f"[weights] 下载失败：{e}\n提示：可改用离线权重包，或稍后重试（断点续传）")
        sys.exit(1)
