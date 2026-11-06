#
# Reproduce the "switch JSON专家模式 -> 图形输入模式 breaks sliders" bug.
# Two layers:
#   1. offline: call _sync_mode directly
#   2. live: launch the Blocks (no model load) and fire the real queue event
#      through gradio_client, exercising the exact HTTP contract the browser uses
# Run: vera_venv\Scripts\python tests\repro_mode_switch.py
#
import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # keep import order similar to app.py

import app as app_mod  # noqa: E402
import gradio as gr  # noqa: E402
from ui.params_def import (  # noqa: E402
    SLIDER_KEYS,
    json_to_sliders,
    layout_order_keys,
    sliders_to_json,
)

MODE_SLIDERS = "图形输入模式"
MODE_JSON = "JSON 专家模式"


def make_args():
    return argparse.Namespace(
        dtype="auto", model_path="./checkpoints/VeraRetouch",
        config_add_path="./configs/infer_config.yaml",
        pretrained_path="./checkpoints/encoder_renderer.pth",
        model_name="RetouchRenderer_Resnet18Encoder_InputCatMixedCBAM",
        temperature=0.2, top_p=0.8, num_beams=1, max_new_tokens=4096,
        chunk=262144, output_dir="./gradio_outputs_test",
    )


def capture_build():
    """Re-create build_ui internals can't be called directly; build demo and
    locate the mode-radio change event + extract the sync handler offline."""
    runner = app_mod.VeraRetouchRunner(make_args())
    ref_runner = app_mod.ReferenceRetouchRunner(make_args())
    demo = app_mod.build_ui(runner, ref_runner, make_args())
    return demo, runner, ref_runner


def offline_sync_via_like_handler():
    """Replicate _sync_mode logic offline (protocol<->layout mapping)."""
    from ui.params_def import layout_order_keys
    layout = layout_order_keys()
    # simulate: user set exposure=30 & colortemp=-40 in JSON mode by hand json
    hand_json = '{"Exposure2012":{"value":30},"IncrementalTemperature":{"value":-40}}'
    parsed, unknown = json_to_sliders(hand_json)
    print(f"[offline] parsed ok: {parsed[:3]}... unknown={unknown}")


def live_test():
    demo, runner, ref_runner = capture_build()
    demo.queue(max_size=8)
    demo.launch(prevent_thread_lock=True, quiet=True, server_name="127.0.0.1", server_port=7905)
    port = 7905

    # find the mode_radio change event (40 inputs / 41 outputs)
    target = None
    for i, fn in enumerate(demo.fns):
        try:
            n_in = len(fn.get_config()["inputs"]) if hasattr(fn, "get_config") else None
        except Exception:
            n_in = None
    # demo.fns entries expose config via blocks_events? fallback: inspect via demo config
    cfg = demo.get_config_file() if hasattr(demo, "get_config_file") else None
    # Use public config json from server instead
    import requests
    config = requests.get(f"http://127.0.0.1:{port}/config").json()
    # find radio change dependencies
    dep_target, fn_index = None, None
    comps = {c["id"]: c for c in config["components"]}
    radio_id = next(c["id"] for c in config["components"]
                    if c["type"] == "radio" and "参数输入方式" in str(c.get("props", {}).get("label", "")))
    for dep in config["dependencies"]:
        if dep.get("targets") == [[radio_id, "change"]] or (
                isinstance(dep.get("targets"), list) and dep.get("targets")
                and dep["targets"][0][1] == "change" and dep["targets"][0][0] == radio_id):
            fn_index = dep["id"]
            dep_target = dep
            break
    assert dep_target is not None, "radio change event not found"
    print(f"[live] radio-change dependency id={fn_index}, "
          f"inputs={len(dep_target['inputs'])}, outputs={len(dep_target['outputs'])}")

    from gradio_client import Client
    client = Client(f"http://127.0.0.1:{port}/")
    zeros = [0] * 38
    # 1) 图形输入模式 default -> switch to JSON mode
    print("[live] switch to JSON mode ...")
    res = client.predict(
        MODE_JSON, sliders_to_json(*zeros), *zeros,
        api_name=f"/_sync_mode" if False else None,
        fn_index=fn_index,
    )
    print("[live] -> JSON mode OK", type(res))

    # 2) switch BACK to slider mode (the failing path in the bug report)
    print("[live] switch back to slider mode ...")
    res = client.predict(
        MODE_SLIDERS, sliders_to_json(*zeros), *zeros,
        fn_index=fn_index,
    )
    print("[live] -> slider mode OK", type(res))


if __name__ == "__main__":
    try:
        offline_sync_via_like_handler()
        live_test()
        print("REPRO DONE (no failure captured)")
    except Exception:
        traceback.print_exc()
    finally:
        # ensure server shutdown happens when script exits with prevent_thread_lock
        pass
