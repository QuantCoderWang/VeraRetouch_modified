#
# Parameter definitions for the graphical (slider) input mode.
# The 38 keys and their order intentionally match data_samples/param.json and
# data/infer_dataset.Infer_Param_Dataset.get_organized_dict():
#   - values on the sliders are 0..100-scale (divided by 100 on submit)
#   - group mismatch note: if every slider of one group stays at 0, the model
#     masks that control latent entirely (get_mask_from_path semantics).
#
import json

# (group_id, Chinese group title, default accordion state)
GROUP_LIGHT = 1
GROUP_COLOR = 2
GROUP_HSL = 3

_COLORS = [("Red", "红"), ("Orange", "橙"), ("Yellow", "黄"), ("Green", "绿"),
           ("Aqua", "青"), ("Blue", "蓝"), ("Purple", "紫"), ("Magenta", "洋红")]

# {key: {"name": 中文名, "desc": 作用说明, "group": 1|2|3}}
PARAM_DEFS = {
    # --- 第 1 组 · 光线调整 ---
    "Exposure2012": {"name": "曝光度", "desc": "整体提亮/压暗画面。正值增加曝光使照片变亮；负值减少曝光、变暗", "group": GROUP_LIGHT},
    "Contrast2012": {"name": "对比度", "desc": "增强画面明暗反差。正值对比更强、更\"硬\"；负值更柔和、发灰", "group": GROUP_LIGHT},
    "Highlights2012": {"name": "高光", "desc": "单独调整画面最亮区域。负值找回过曝高光细节；正值高光更突出", "group": GROUP_LIGHT},
    "Shadows2012": {"name": "阴影", "desc": "单独调整画面最暗区域。正值提亮暗部、找回阴影细节；负值暗部更沉", "group": GROUP_LIGHT},
    "Whites2012": {"name": "白色色阶", "desc": "调整接近纯白的像素。正值白场更通透；负值抑制过白的死白", "group": GROUP_LIGHT},
    "Blacks2012": {"name": "黑色色阶", "desc": "调整接近纯黑的像素。负值黑场加深、画面更\"实\"；正值黑场发灰", "group": GROUP_LIGHT},
    "ParametricShadows": {"name": "参数曲线 · 阴影区", "desc": "类 PS 曲线中的阴影分段。正值提亮阴影段；负值压暗", "group": GROUP_LIGHT},
    "ParametricDarks": {"name": "参数曲线 · 暗调区", "desc": "类 PS 曲线中的偏暗分段。正值提亮；负值压暗", "group": GROUP_LIGHT},
    "ParametricLights": {"name": "参数曲线 · 亮调区", "desc": "类 PS 曲线中的偏亮分段。正值提亮；负值压暗", "group": GROUP_LIGHT},
    "ParametricHighlights": {"name": "参数曲线 · 高光区", "desc": "类 PS 曲线中的高光分段。正值提亮；负值压暗", "group": GROUP_LIGHT},
    # --- 第 2 组 · 色温与整体色彩 ---
    "IncrementalTemperature": {"name": "色温（冷暖）", "desc": "正值偏黄暖调；负值偏蓝冷调", "group": GROUP_COLOR},
    "IncrementalTint": {"name": "色调（绿↔品红）", "desc": "正值偏品红；负值偏绿。常用于校正或强化环境色", "group": GROUP_COLOR},
    "Vibrance": {"name": "自然饱和度", "desc": "智能加饱和：优先增强低饱和色，肤色不易过火", "group": GROUP_COLOR},
    "Saturation": {"name": "饱和度", "desc": "全体颜色饱和度平移。正值更艳；负值趋近黑白", "group": GROUP_COLOR},
}
# --- 第 3 组 · 特定颜色调整（HSL）：色相 8 项 → 饱和 8 项 → 明度 8 项 ---
for _color_key, _color_cn in _COLORS:
    PARAM_DEFS[f"HueAdjustment{_color_key}"] = {
        "name": f"色相 · {_color_cn}",
        "desc": f"{_color_cn}色区域色相偏移（按色相轮方向：正/负分别偏向相邻色）",
        "group": GROUP_HSL,
    }
for _color_key, _color_cn in _COLORS:
    PARAM_DEFS[f"SaturationAdjustment{_color_key}"] = {
        "name": f"饱和 · {_color_cn}",
        "desc": f"只增减{_color_cn}色的浓淡程度。正值更艳；负值减淡",
        "group": GROUP_HSL,
    }
for _color_key, _color_cn in _COLORS:
    PARAM_DEFS[f"LuminanceAdjustment{_color_key}"] = {
        "name": f"明度 · {_color_cn}",
        "desc": f"只提亮/压暗{_color_cn}色的亮度，不改变颜色倾向",
        "group": GROUP_HSL,
    }

# Canonical slider order (= param.json order: light, color, hue*, sat*, lum*)
SLIDER_KEYS = list(PARAM_DEFS.keys())
assert len(SLIDER_KEYS) == 38, PARAM_DEFS

SLIDER_MIN, SLIDER_MAX, SLIDER_STEP = -100, 100, 1

UI_SLIDER_COLUMNS = 2  # 每组滑块在界面中的列数


def layout_order_keys():
    """Slider creation order under the two-column UI layout.

    The param tab lays each group out in `UI_SLIDER_COLUMNS` columns placed
    left-column-first, so the positional order of gr.Slider components is NOT
    the protocol (SLIDER_KEYS) order. app.py builds sliders following this
    function and uses it as the authority for positional <-> key mapping.
    """
    out = []
    ncols = UI_SLIDER_COLUMNS
    for group_id in (GROUP_LIGHT, GROUP_COLOR, GROUP_HSL):
        group_keys = [k for k in SLIDER_KEYS if PARAM_DEFS[k]["group"] == group_id]
        cols = [group_keys[i::ncols] for i in range(ncols)]
        for col in cols:
            out.extend(col)
    return out


def default_slider_values():
    """All-zero start state (graphic mode default = 未启用 any group)."""
    return [0] * len(SLIDER_KEYS)


def sliders_to_json(*slider_values):
    """Pack slider values into the pipeline JSON format {"Key":{"value":v}}."""
    if len(slider_values) != len(SLIDER_KEYS):
        raise ValueError(f"expected {len(SLIDER_KEYS)} values, got {len(slider_values)}")
    payload = {}
    for key, value in zip(SLIDER_KEYS, slider_values):
        payload[key] = {"value": int(round(float(value)))}
    return json.dumps(payload, ensure_ascii=False, indent=4)


def json_to_sliders(text):
    """Parse pipeline JSON -> (38 slider ints, list of unknown keys). Ignores missing keys -> 0."""
    data = json.loads(text or "")
    if not isinstance(data, dict):
        raise ValueError("JSON 顶层必须是对象 {\"参数键名\":{\"value\":x}}")
    values, unknown = [], []
    for key in SLIDER_KEYS:
        entry = data.get(key)
        if isinstance(entry, dict) and "value" in entry:
            try:
                v = int(round(float(entry["value"])))
            except (TypeError, ValueError):
                raise ValueError(f"参数 {key} 的 value 不是数字")
            v = max(SLIDER_MIN, min(SLIDER_MAX, v))
        else:
            v = 0
        values.append(v)
    unknown = [k for k in data.keys() if k not in PARAM_DEFS]
    return values, unknown


def values_from_param_file(path):
    """Read a shipped param.json -> 38 slider ints (used by 载入示例参数)."""
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    values = []
    for key in SLIDER_KEYS:
        entry = data.get(key, {})
        try:
            v = int(round(float(entry.get("value", 0)))) if isinstance(entry, dict) else 0
        except (TypeError, ValueError):
            v = 0
        values.append(max(SLIDER_MIN, min(SLIDER_MAX, v)))
    return values
