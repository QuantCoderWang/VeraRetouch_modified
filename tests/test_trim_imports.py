#
# Verify the inference chain imports cleanly when matplotlib / skimage are
# ABSENT (simulating the trimmed pack environment), including the lazily
# imported data/data_utils.py module.
#
import builtins
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

blocked = {"matplotlib", "skimage"}
real_import = builtins.__import__


def guard(name, *args, **kwargs):
    root = name.split(".")[0]
    if root in blocked:
        raise ModuleNotFoundError(f"BLOCKED for test: {root}")
    return real_import(name, *args, **kwargs)


builtins.__import__ = guard
try:
    import gradio, cv2, transformers, timm, diffusers, yaml  # noqa: E401,F401
    import data.data_utils        # 惰性导入兜底验证
    import data.infer_dataset     # 推理链
    import utils                  # renderer 入口
    import app                    # Gradio 界面入口
    import inference              # CLI 入口
    print("[OK] full import chain works without matplotlib/skimage")
    builtins.__import__ = real_import
    sys.exit(0)
except ModuleNotFoundError as e:
    print(f"[FAIL] {e}")
    sys.exit(1)
finally:
    builtins.__import__ = real_import
