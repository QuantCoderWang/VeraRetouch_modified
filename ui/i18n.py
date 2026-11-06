#
# Centralized Chinese UI copy for the VeraRetouch Gradio interface.
# Per 方案 §2: only display strings live here — parameter keys stay English
# because the inference pipeline matches on them.
#

TITLE = "VeraRetouch 修图工具"
PAGE_HEADER = "VeraRetouch 修图工具"
PAGE_SUBHEADER = "模型随程序一次性加载并常驻内存"
DEVICE_NOTE = "当前运行设备：{device}"
CPU_NOTE = "（CPU 模式，单图约需 2–5 分钟，请耐心等待进度条）"

TAB_AUTO = "一键自动修图"
TAB_STYLE = "风格修图"
TAB_PARAM = "参数修图"
TAB_REF = "参考图修图"

BTN_START_AUTO = "开始修图"
BTN_START_STYLE = "开始风格修图"
BTN_START_PARAM = "开始参数修图"
BTN_START_REF = "开始参考图修图"

INPUT_IMAGE_LABEL = "输入图片"
INPUT_IMAGE_INFO = "支持 jpg / png"
OUTPUT_IMAGE_LABEL = "修图结果"
ANALYSIS_LABEL = "模型分析与修图计划（由模型生成，原文为英文）"

STYLE_PROMPT_LABEL = "风格描述"
STYLE_PROMPT_PLACEHOLDER = "I want a dreamy bright pink style."
STYLE_PROMPT_INFO = "风格描述建议输入英文，效果更佳；例如：想要梦幻明亮的粉色调"

REF_INFO_MD = (
    "上传输入图片与两张参考图（**三张都必填**）。"
    "**参考图 · 修图前**：参考素材的原始状态；"
    "**参考图 · 修图后**：参考素材的期望修图效果。"
    "模型将学习两张参考图之间的修图风格迁移到你的输入图片上。"
)
REF_INPUT_LABEL = "输入图片（必填）"
REF_BEFORE_LABEL = "参考图 · 修图前（必填）"
REF_AFTER_LABEL = "参考图 · 修图后（必填）"

# --- 参数修图页 ---
# 参数输入只保留图形（滑块）模式；JSON 专家模式已于 2026-11-05 应用户反馈移除。
# （内部滑块→JSON 转换仍在使用：模型管线需要 param.json 文件，见 app.py）
PARAM_GROUP_TITLES = {1: "光线调整", 2: "色温与整体色彩", 3: "特定颜色调整（HSL）"}
PARAM_SLIDER_RANGE_INFO = "取值范围 -100 ~ 100，提交时内部 ÷100 归一化为 -1.0 ~ 1.0"
BTN_RESET_ALL = "全部归零"
BTN_LOAD_SAMPLE = "载入示例参数"

# --- 导出格式 ---
EXPORT_FORMAT_LABEL = "导出格式"
EXPORT_FORMAT_INFO = (
    "「跟随输入格式」＝输出文件与上传图片同格式（识别不了时自动用 PNG）；"
    "JPG / WebP 固定质量 95。结果预览与下载文件就是实际保存的格式，不再被网页转成 webp。"
)
EXPORT_FMT_AUTO = "跟随输入格式"
EXPORT_FMT_PNG = "PNG（无损）"
EXPORT_FMT_JPG = "JPG（体积小，质量 95）"
EXPORT_FMT_WEBP = "WebP（体积最小，质量 95）"
EXPORT_FORMAT_CHOICES = [
    (EXPORT_FMT_AUTO, "auto"),
    (EXPORT_FMT_PNG, "png"),
    (EXPORT_FMT_JPG, "jpg"),
    (EXPORT_FMT_WEBP, "webp"),
]

# --- 批量自动修图页（2026-11-05 新增，见 docs/批量自动修图_技术方案.md） ---
TAB_BATCH = "批量自动修图"
BATCH_INFO_MD = (
    "选择**待修图文件夹**与**输出文件夹**即可全自动修图：每张图自动保存修图结果"
    "（**跟随输入格式**）与**同名分析记录 txt**（该图的模型分析与修图计划）；"
    "输出同名文件已存在时自动跳过（中断后重跑可续）；**单张失败不影响整批**。"
)
BATCH_IN_DIR_LABEL = "待修图文件夹（必填）"
BATCH_OUT_DIR_LABEL = "输出文件夹（留空则自动写入：输入文件夹\\veraretouch_批修输出）"
BATCH_PICK_TITLE_IN = "选择待修图文件夹"
BATCH_PICK_TITLE_OUT = "选择输出文件夹"
BTN_BROWSE = "浏览…"
BTN_SCAN = "扫描图片"
BTN_START_BATCH = "开始批量修图"
BATCH_SCAN_FMT = "共找到 **{n}** 张：jpg/jpeg {a} · png {b} · webp {c} · bmp/tif 其他 {d}"
BATCH_SCAN_EMPTY = "该文件夹内没有可处理的图片（支持 jpg / jpeg / png / webp / bmp / tif / tiff）"
BATCH_SCAN_MISS = "路径不存在或不可读：{path}"
BATCH_ERR_NO_INDIR = "待修图文件夹不存在或不可读：{path}"
BATCH_ERR_NO_IMG = "文件夹中没有可处理的图片（支持 jpg / jpeg / png / webp / bmp / tif / tiff）"
BATCH_ERR_SAME_DIR = "输出文件夹不能与待修图文件夹相同：无法区分原文件与处理结果，请另选一个输出文件夹"
BATCH_OVERALL_FMT = "第 {i} / {n} 张 · 正在处理 {name} · 预计剩余 {eta}"
BATCH_LOG_HEADER = "==== 批量修图汇总（同时保存到输出文件夹 batch_summary.txt）===="
BATCH_IMG_FAILED_WARN = "{name} 处理失败，已记录，继续处理其余图片"
BATCH_DONE = "批量修图完成：成功 {ok} 张 · 失败 {fail} 张 · 跳过 {skip} 张，输出目录：{dir}"
BATCH_SUMMARY_LABEL = "批量完成汇总（实时滚动）"
# 汇总滚动框上方的实时统计行（用户 2026-11-05 反馈："去掉预览图与最近一张分析框，只留汇总"）
BATCH_STATS_FMT = "🗂 **处理总量 {total}** ｜ ✅ 已成功 **{ok}** ｜ ❌ 失败 **{fail}** ｜ ⏭ 跳过 **{skip}**"
BATCH_STATS_CUR_FMT = "　⏳ 正在处理：**{name}**"

# --- 输出降噪（方案一·细节回贴，2026-11-06）：NLM h=10 强降噪后，按档位把原图高频
#     细节层贴回去 final = dn + α×(orig−dn)；默认关（模型裸输出零改动）---
DENOISE_LABEL = "输出降噪"
DENOISE_OFF = "关闭（保留全部细节）"
DENOISE_LIGHT = "轻（细节优先）"
DENOISE_MEDIUM = "中（均衡）"
DENOISE_STRONG = "强（画面最干净）"
DENOISE_CHOICES = [(DENOISE_OFF, "off"), (DENOISE_LIGHT, "light"),
                   (DENOISE_MEDIUM, "medium"), (DENOISE_STRONG, "strong")]
DENOISE_INFO = (
    "修图提亮暗部会让照片原有的传感器噪点被放大。降噪后会把发丝/织物等高频细节"
    "按比例贴回，档位越高越干净、细节牺牲越多；默认关闭，输出与模型结果完全一致。"
)

# --- 进度文案 ---
PROGRESS_PREPROCESS = "正在预处理图片…"
PROGRESS_TEXTGEN_FMT = "文本生成中：已生成 {count} 个 token…（CPU 上最耗时步骤）"
PROGRESS_TEXTGEN_START = "文本生成开始（CPU 上最耗时步骤）…"
PROGRESS_RENDER = "正在渲染修图结果…"
PROGRESS_REF = "正在运行参考图修图（Encoder-Renderer）…"
PROGRESS_DONE = "完成"

# --- 报错 ---
ERR_NEED_IMAGE = "请先上传图片"
ERR_NEED_STYLE = "请输入风格描述（建议英文，效果更佳）"
ERR_REF_NEED_INPUT = "请上传输入图片后再开始参考图修图"
ERR_REF_NEED_BEFORE = "请上传「参考图 · 修图前」：该功能通过比较参考图的修图前后变化来学习风格，缺少追溯无法运行"
ERR_REF_NEED_AFTER = "请上传「参考图 · 修图后」：该功能需要一张期望效果参考图，否则无法运行"
ERR_REF_MISSING = "请上传输入图片与参考图（修图后）"  # 保留兼容旧引用；新代码使用上面三条分项报错
ERR_GENERATION_FMT = (
    "生成失败：{error}。模型偶发漏生成修图 token 时会触发该问题；"
    "请直接重试一次（temperature > 0 时每次结果不同）。"
)
WARN_ALL_ZERO_SUBMIT = "当前参数全为 0：模型将不注入任何参数控制信号（三组都不会生效），结果近似自动修图。确认继续吗？"
