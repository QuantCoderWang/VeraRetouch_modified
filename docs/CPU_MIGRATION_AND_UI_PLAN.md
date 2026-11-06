# VeraRetouch CPU 移植与图形界面方案

> 状态：**✅ 已实施并验证通过（2026-11-04）**
> 分析日期：2026-11-04
> 目标：在无独立 GPU 的 Windows 电脑（Python 3.12）上运行 VeraRetouch 全部推理功能，并补充图形界面

## ✅ 实施结果摘要

| 项目 | 结果 |
|---|---|
| 依赖安装（Python 3.12 venv，torch 2.15.0+cpu） | ✅ 完成 |
| 模型权重下载（主模型 2.29GB + encoder_renderer 182MB，HF 镜像） | ✅ 完成 |
| `inference.py` CPU 化（设备自适应 + `--dtype` + num_workers=0 + **use_cache=True 修复**） | ✅ 完成 |
| `infer_ref_retouch.py`（map_location） | ✅ 完成 |
| 四条命令行链路（ref/auto/style/param） | ✅ 全部通过 |
| Gradio 界面 `app.py`（四页签） | ✅ 完成，四个接口全部端到端验证通过 |

**CPU 实测耗时**（sample_flower.jpg，1707×1280，fp32，15 线程）：
- 参考图修图：~4 秒 ｜ 参数修图：~20 秒 ｜ 风格修图：~65 秒 ｜ 自动修图：~100 秒（含界面调用开销）

**实施中的关键发现与修复**（方案阶段未预见的）：
1. **`use_cache=false` 陷阱**：checkpoint 的 config.json 出厂即 `use_cache: false`（作者训练配置遗留），生成时不启用 KV 缓存会导致 CPU 上每 token 重算全前缀、速度慢一个数量级。已在 `inference.py` 与 `app.py` 的生成调用中显式传 `use_cache=True`；
2. **CPU 渲染器内存**：不分块时渲染 1707×1280 图需一次性分配 4.47GB 激活内存（会 OOM）。`app.py` 在 CPU 下自动启用 `chunk=262144`；
3. **参数页签传参错位 bug**：Gradio 按 inputs 位置映射 handler 参数，原统一签名导致 param JSON 落入 prompt 参数。已改为每模式独立 handler（auto/style/param/ref 四个具名函数）。

**已知行为说明**：`_generate` 中 LLM 偶发漏生成某个 retouch token 时会抛索引异常（原项目自身行为，CLI 端原代码用 try/except 吞掉）。app.py 已将其包装为友好报错提示重试。

---


---

## 一、项目分析结论

### 1.1 架构组成（已从代码逐层验证）

本项目是 LLaVA 风格的多模态修图框架，核心组件：

| 组件 | 实现 | 规模/来源 |
|---|---|---|
| 主模型 `VeraRetouchForCausalLLM_Unified` | Qwen2-0.5B LLM（hidden_size=896）+ MobileCLIP-L/FastViT-HD 视觉塔 + `retouch_head`(MLP) + `retouch_decoder`(ResNet18 渲染器) | HuggingFace `Gyh68/VeraRetouch`，**2.29 GB** safetensors |
| 参考图修图模型 `RetouchRenderer` | ResNet18 编码器 + ConditionalMLPDecoder，独立于主模型 | HuggingFace `Gyh68/VeraRetouch.Encoder_Renderer` 的 `encoder_renderer.pth`（较小） |
| 两种推理入口 | `inference.py`（auto/style/param 三模式）、`infer_ref_retouch.py`（参考图修图） | — |

**关键事实：**
- 视觉塔（MobileCLIP）配置从本地 JSON 加载（`llava/model/multimodal_encoder/mobileclip/configs/mobileclip_l.json`），权重包含在主 checkpoint 内，**无需额外下载**；
- 默认渲染器 `RetouchRenderer_Resnet18Encoder_InputCatMixedCBAM` 为纯 ResNet18 结构（`configs/renderer_config.py:274`），无外部依赖；
- 推理导入链（`inference.py` → `llava/*` → `model/*`）**不含** flash_attn / deepspeed / xformers / sglang / bitsandbytes —— deepspeed 仅出现在训练函数体内部（`llava/train/train_qwen_comment.py:138`，`maybe_zero_3` 内），推理路径不会触发，注释掉这些 GPU 依赖是安全的；
- 生成流程：图片 → 视觉编码 → LLM 生成"问题分析 + 修图计划"文本与 3 个特殊 token（`<retouch_light>` 等）→ 取这些 token 位置的隐藏层向量 → `retouch_head` 映射为控制潜变量 → 渲染器输出修图结果。**LLM 每图生成数百 token，是 CPU 耗时主体。**

### 1.2 CPU 可行性评估：✅ 可行

- **内存**：权重 bf16 约 2.3 GB；转 fp32 约 4.6 GB，加激活与 KV cache，**建议 16 GB 内存**（8 GB 机器可用 bf16 运行主模型，风险后述）；
- **速度**：Qwen2-0.5B 在主流桌面 CPU 上约 5~15 token/s，单图总耗时预估 **2~5 分钟**（视觉编码数秒 + 文本生成 1~3 分钟 + 渲染数秒）。可用但无实时性，UI 需明示"处理中"状态；
- 渲染器与视觉塔均为轻量卷积网络，CPU 开销可忽略。

### 1.3 界面现状：❌ 无可用界面，需要新建

- `llava/serve/gradio_web_server.py` 是 **LLaVA 原版聊天服务器**：需要先启动 `controller.py` + `model_worker.py` 两个服务，且只支持文本对话，**不支持本项目的修图图片输出**，不能直接复用；
- `ml-veraretouch/` 是 **macOS/iOS 的 Swift 工程**（Xcode 项目 + CoreML/MLX），与 Windows 无关；
- 结论：需新建一个 Gradio Web UI。你替换的 requirements.txt 已包含 `gradio==6.14.0`、`fastapi`、`uvicorn`（依赖自洽）。

### 1.4 环境实测结果（本次分析中实际执行了检查）

| 检查项 | 结果 |
|---|---|
| `vera_venv`（Python 3.12.10） | **当前是空的，仅含 pip 25.0.1** —— 任何包都还没安装 |
| pip 镜像 | 已配置阿里云镜像（mirrors.aliyun.com）✅ |
| requirements.txt 依赖解析（`pip install --dry-run -r`） | **全部可解析**；`peft==0.15.0` 会自动带入 **torch 2.15.0（Windows PyPI 版即 CPU 版）+ torchvision 0.29.2**，无需单独装 torch |
| `checkpoints/` 目录 | **不存在** —— 需从 HuggingFace 下载 2.29 GB 权重（本次分析中 huggingface.co 直连超时，需用镜像，见阶段 0） |
| 样例数据 | `data_samples/input/sample_flower.jpg`、`ref_inputs/`、`param.json` 齐全 ✅ |

---

## 二、需要修改的问题清单

### 2.1 GPU 硬编码（阻塞 CPU 运行的必改项）

| # | 位置 | 现状 | 问题 |
|---|---|---|---|
| G1 | `inference.py:58` | `VeraRetouchForCausalLLM_Unified.from_pretrained(..., torch_dtype=torch.bfloat16).cuda()` | `.cuda()` 在无 CUDA 机器直接抛 `AssertionError`/`RuntimeError` |
| G2 | `inference.py:95` | `batch['input_ids'].cuda()` | 同上 |
| G3 | `inference.py:97` | `[tensor.to(torch.bfloat16).cuda() for tensor in images]` | 同上 |
| G4 | `inference.py:98` | `attention_mask.cuda()` | 同上 |
| G5 | `inference.py:100` | `[tensor.to(torch.bfloat16).cuda() for tensor in input_imgs]` | 同上 |
| G6 | `inference.py:103` | `[tensor.to(torch.bfloat16).cuda() for tensor in retouch_masks]` | 同上 |
| G7 | `infer_ref_retouch.py:44` | `torch.load(args.pretrained_path)` 无 `map_location` | 若 checkpoint 含 CUDA 张量，CPU 机器上 `torch.load` 直接失败（该文件其余部分已有 `cuda if available else cpu` 的回退，唯独漏了这一处） |

### 2.2 dtype 策略（影响 CPU 速度与兼容性）

- `inference.py:57` 固定 `torch_dtype=torch.bfloat16`。PyTorch CPU 支持 bf16，但**老 CPU 无 AVX512-BF16/AMX 指令时速度显著劣于 fp32**，且部分算子组合在 CPU bf16 下数值不稳定。方案：改为 `--dtype` 参数，**CPU 下默认 float32**，允许手动切 bf16 省内存。

### 2.3 Windows / 版本兼容隐患（启动验证时需重点观察）

| # | 风险点 | 说明与对策 |
|---|---|---|
| C1 | **opencv-python==4.8.0.76 + numpy==2.4.4 ABI 冲突** | 4.8.0.76（2023 年）按 numpy 1.x ABI 编译，numpy 2.x 下 `import cv2` 会报 *"A module compiled with NumPy 1.x cannot run in NumPy 2.x"*。pip 能装上但**运行必崩**。对策：requirements.txt 中升级为 `opencv-python>=4.10.0.84`（首个 numpy2 构建版本），功能 API 完全兼容 |
| C2 | `inference.py:85` `num_workers=8` | Windows DataLoader 用 spawn，8 个子进程反复重导入 llava/torch，启动慢且可能因对象序列化失败报错。对策：CPU 下改为 `0` |
| C3 | transformers 4.57.1 的 `torch_dtype` 参数已更名 `dtype`（旧名仍兼容但发 FutureWarning） | 若报 `unexpected keyword 'torch_dtype'`，改用 `dtype=` |
| C4 | `from_pretrained(..., config_add=...)` 向自定义 `__init__` 透传自定义参数 | 依赖 transformers 内部 kwargs 转发行为，4.57 实测如有异常，备选方案：手动 `LlavaConfig.from_pretrained()` → `VeraRetouchForCausalLLM_Unified(config, config_add)` → `load_state_dict(safetensors.load_file(...))`（方案文档阶段先记录，验证时确认） |
| C5 | torch≥2.6 默认 `torch.load(weights_only=True)` | `encoder_renderer.pth` 是 `{"model_state_dict": ...}` 字典，若含非张量对象会报错。对策：`torch.load(..., map_location=device)` 报错时显式加 `weights_only=False`（本仓库自家权重，来源可信） |
| C6 | HuggingFace 直连超时（已实测） | 下载 2.29 GB 权重需走 `HF_ENDPOINT=https://hf-mirror.com` 镜像 |

---

## 三、分阶段实施计划

### 阶段 0：环境准备（不改动项目代码）

```bash
# 1) 修正 requirements.txt 中一行（见 C1）：
#    opencv_python==4.8.0.76  →  opencv_python==4.10.0.84

# 2) 安装全部依赖（torch 2.15.0 CPU 版会随 peft 自动装入，无需单独装）
D:\VeraRetouch\VeraRetouch-main\vera_venv\Scripts\pip.exe install -r requirements.txt

# 3) 下载主模型权重（国内镜像）
set HF_ENDPOINT=https://hf-mirror.com
D:\VeraRetouch\VeraRetouch-main\vera_venv\Scripts\huggingface-cli.exe download Gyh68/VeraRetouch --local-dir ./checkpoints/VeraRetouch

# 4)（可选，参考图修图功能需要）下载 encoder_renderer.pth
D:\VeraRetouch\VeraRetouch-main\vera_venv\Scripts\huggingface-cli.exe download Gyh68/VeraRetouch.Encoder_Renderer encoder_renderer.pth --local-dir ./checkpoints

# 5) 冒烟测试：验证导入链在 CPU 环境可完整加载（不含模型权重）
vera_venv\Scripts\python.exe -c "import torch, cv2, transformers, timm, diffusers, gradio; import inference" 
#（注：import inference 会执行模块级导入，不触发 __main__；若报 ModuleNotFoundError 逐一排查）
```

**验收标准**：第 5 步无报错，且 `torch.cuda.is_available() == False`、`import cv2` 正常。

### 阶段 1：核心代码 CPU 化（最小侵入修改，共 2 个文件）

#### 1.1 `inference.py`

```python
# (a) predict() 开头新增设备与精度选择（替换 G1–G6 的散落硬编码）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = {"cuda": torch.bfloat16, "cpu": torch.float32}[device.type]   # CPU 默认 fp32
torch.set_num_threads(max(1, os.cpu_count() - 1))                      # CPU 线程利用

# (b) L53-58 模型加载：去掉 .cuda()，dtype 可配置
model = VeraRetouchForCausalLLM_Unified.from_pretrained(
    model_path, config_add=config_add, cache_dir='./cache',
    torch_dtype=dtype,          # 若 transformers 4.57 报参数名错误则改 dtype=dtype（C3）
).to(device)                    # ← 原 .cuda()

# (c) L95-103 批次数据：统一 .to(device)，图像张量用 dtype
input_ids = batch['input_ids'].to(device)
images = [t.to(device=device, dtype=dtype) for t in batch.get('images', []) or []]
attention_mask = batch['attention_mask'].to(device)
input_imgs = [t.to(device=device, dtype=dtype) for t in batch['input_imgs']]
retouch_masks = [t.to(device=device, dtype=dtype) for t in batch['retouch_masks']]

# (d) L85 DataLoader：Windows 下关闭多进程加载
num_workers=0                   # ← 原 8（C2）

# (e) 参数区新增 --dtype {auto,float32,bfloat16}（默认 auto：GPU→bf16，CPU→fp32）
```

> 注：`retouch_masks` 是 `[1.0,1.0,1.0]` 掩码，实际用 fp32/bf16 均可，与 latent 相乘时类型需一致，故统一用 `dtype`。

#### 1.2 `infer_ref_retouch.py`

```python
# L44：加 map_location（G7）；若触发 weights_only 报错再补 weights_only=False（C5）
model.load_state_dict(
    torch.load(args.pretrained_path, map_location=device)["model_state_dict"]
)
```

该文件其余部分（`device = 'cuda' if available else 'cpu'`，L40）已就绪，无需改动。

#### 1.3 不改动的部分（明确说明）

- `llava/`、`model/`、`data/` 全部模型与数据代码：**零修改**（已验证无 CUDA 依赖）；
- `inference.py:40-43` 的 `generation_config.json` 临时重命名逻辑：HF 仓库中该文件本身就是隐藏名 `.generation_config.json`，此逻辑不会触发，无影响；
- 训练相关文件（`llava/train/*`）：不进入推理路径，保持原样。

### 阶段 2：命令行启动验证（按序执行，每步给出预期）

```bash
# 步骤 1：参考图修图（模型小、加载快，最先验证依赖链）
python infer_ref_retouch.py --pretrained_path ./checkpoints/encoder_renderer.pth \
    --output_dir ./data_samples/ref_outputs \
    --input_img_path ./data_samples/ref_inputs/sample.jpg \
    --ref_before_img_path ./data_samples/ref_inputs/sample.jpg \
    --ref_after_img_path ./data_samples/ref_inputs/ref/after.jpg
# 预期：终端打印 "Device: cpu"，数十秒内输出 ref_outputs/sample.png

# 步骤 2：主模型 auto 模式（小图 + chunk 分块，控制内存）
python inference.py --mode auto --model-path ./checkpoints/VeraRetouch \
    --img_paths ./data_samples/input/sample_flower.jpg \
    --save_dir ./data_samples/output/ --chunk 262144 --batch_size 1
# 预期：加载约 1-2 分钟，生成约 2-5 分钟；输出 output/pred_imgs/sample_flower.png + texts/*.txt

# 步骤 3：style 模式（追加 --prompt "I want a dreamy bright pink style."）
# 步骤 4：param 模式（追加 --instruction_path ./data_samples/param.json）
```

**验证期已知风险对照表**（出现即按对策处理）：C3/C4/C5 见上表；若内存不足（机器 <16 GB）：`--dtype bfloat16` 重试；大图务必带 `--chunk 262144`。

### 阶段 3：新增 Gradio 图形界面（新文件 `app.py`，约 300 行）

**设计原则**：模型**进程内常驻、只加载一次**（避免每张图重载 2.29 GB）；单并发（`gr.Blocks` + 队列）；界面明示 CPU 耗时预期。

**功能布局（4 个页签）：**

```
┌────────────────────────────────────────────────────────┐
│ VeraRetouch Web UI（标题 + 模型加载状态徽标）              │
├─ Tab1 自动修图 ────────────────────────────────────────┤
│  [上传图片]                          [开始修图按钮]       │
│  ┌──────────────┬──────────────┐                       │
│  │   原图预览    │  修图结果     │   ← 并排对比           │
│  └──────────────┴──────────────┘                       │
│  模型分析文本（问题诊断 + 修图计划，等宽字体滚动区）        │
├─ Tab2 风格修图 ──── 同上 + [风格描述文本框] ─────────────┤
├─ Tab3 参数修图 ──── 同上 + [JSON 参数编辑器(预填 param.json 默认值)] ─┤
├─ Tab4 参考图修图 ── [输入图][参考图-前][参考图-后] → [结果] │
│ （Tab4 使用 encoder_renderer.pth，独立小模型，懒加载）     │
└────────────────────────────────────────────────────────┘
```

**实现要点：**
1. 复用 `Infer_Auto/Style/Param_Dataset` + `DataCollatorForUnifiedTestDataset` 构造单样本批次，直接调 `model._generate(...)`（与 `inference.py` 的 predict 循环一致，输出 PNG + 文本）；
2. Tab4 复用 `infer_ref_retouch.py` 的预处理函数与 `get_model()`；
3. `dtype/device` 逻辑与阶段 1 相同；启动参数：`--model-path`、`--port 7860`、`--share`（可选公网链接）；
4. 长任务用 `gr.Progress` + 状态文本（"视觉编码中… / 文本生成中… / 渲染中…"）；
5. 结果图同时落盘到 `./gradio_outputs/<时间戳>/`，便于回溯。

**启动方式：** `python app.py --model-path ./checkpoints/VeraRetouch` → 浏览器打开 `http://127.0.0.1:7860`

### 阶段 4：性能与体验优化（可选，验证通过后按需做）

| 优化 | 收益 | 成本/风险 |
|---|---|---|
| `torch.set_num_threads(物理核数)` | 已在阶段 1 包含 | 无 |
| `--chunk 262144` 常开 | 内存峰值可控（CPU 无显存，全靠内存） | 速度略降 |
| bf16 运行 | 内存 2.3 GB（省一半） | 老 CPU 上可能更慢，需实测 |
| 文本生成截断：`--max_new_tokens` 适度调小（如 1024） | 减少生成时间 | 可能截断修图 token，需谨慎 |
| torch.compile | CPU 上收益有限 | Windows 支持不稳，**不建议** |
| int8 动态量化（`torch.ao.quantization`） | LLM 部分约 2~3 倍加速 | 精度需校验，工作量中等，**列为后续项** |

---

## 四、实施顺序总结

1. **阶段 0**：修 requirements（opencv 版本）→ 装依赖 → 下载权重（镜像）→ 导入冒烟测试
2. **阶段 1**：改 `inference.py`（7 处 GPU 硬编码 + dtype 可配 + num_workers）与 `infer_ref_retouch.py`（1 处 map_location）
3. **阶段 2**：按"参考图修图 → auto → style → param"顺序命令行验证四条链路
4. **阶段 3**：新建 `app.py` Gradio 界面（四页签，模型常驻）
5. **阶段 4**（可选）：量化等性能优化

**总体结论**：项目 GPU 耦合度低（硬编码集中在 2 个入口文件共 8 处），模型轻量（0.5B LLM），CPU 移植改动小、风险可控；界面缺失属实，用 Gradio 新建即可。预期单图处理 2~5 分钟，满足非实时使用。

---

*本方案仅为分析与设计文档，未对任何源码做修改。实施时请按阶段顺序逐步验证。*
