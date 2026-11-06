# -*- coding: utf-8 -*-
r"""UI 探针启动器（不加载模型权重）：浏览器上传行为验证用。

    vera_venv\Scripts\python.exe tests\_launch_probe.py --port 7899
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as A  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=7899)
args = parser.parse_args()

ns = argparse.Namespace(
    dtype="auto", model_path="./checkpoints/VeraRetouch",
    config_add_path="./configs/infer_config.yaml",
    pretrained_path="./checkpoints/encoder_renderer.pth",
    model_name="RetouchRenderer_Resnet18Encoder_InputCatMixedCBAM",
    temperature=0.2, top_p=0.8, num_beams=1, max_new_tokens=4096,
    chunk=262144, output_dir="./gradio_outputs",
)
ns.chunk = A.adaptive_render_chunk(ns.chunk, A.resolve_device_and_dtype("auto")[0])
runner = A.VeraRetouchRunner(ns)     # 不 load：权重保持未加载，UI 照常渲染
ref_runner = A.ReferenceRetouchRunner(ns)
demo = A.build_ui(runner, ref_runner, ns)
demo.launch(server_name="127.0.0.1", server_port=args.port, inbrowser=False,
            prevent_thread_lock=True)
print("[probe] ready on port", args.port, flush=True)
import time
while True:
    time.sleep(3600)
