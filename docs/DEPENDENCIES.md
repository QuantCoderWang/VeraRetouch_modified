# 依赖包说明（DEPENDENCIES）

> 依赖基线来自原项目 [OpenVeraTeam/VeraRetouch](https://github.com/OpenVeraTeam/VeraRetouch) 的 requirements，并经本项目 CPU 移植与打包精简调整。

本项目 Python **3.12**、Windows 面向。所有依赖版本都经过锁定（`requirements.txt`），并在打包精简环境（`packaging/requirements-pack.txt`）下二次验证过。**普通用户用安装包即可，无需阅读本文**；本文面向从源码运行的开发者。

## ① 从源码运行时需要自己装的包

```powershell
python -m venv vera_venv
vera_venv\Scripts\pip install -r requirements.txt
```

`requirements.txt` 已包含全部锁定版本（共 19 个直接依赖 + 传递依赖约 66 个）。

## ② 直接依赖清单与用途

| 包 | 用途 | 备注 |
|---|---|---|
| **torch / torchvision** | 深度学习框架 | **未写进 requirements.txt，需单独安装**。Windows 下 `pip install torch torchvision` 默认即 CPU 版；GPU 用户按《改造方案》装 CUDA 版 |
| transformers==4.57.1 | 模型加载 / tokenizer / Auto 类 | 主模型 `VeraRetouchForCausalLLM_Unified` 依赖 |
| tokenizers==0.22.1 | 分词器 Rust 后端 | transformers 的性能核心 |
| diffusers==0.34.0 | 渲染管线组件 | `model/retouch_render.py` 渲染器 |
| peft==0.15.0 | LoRA / Adapter | 渲染器权重结构依赖 |
| timm==1.0.26 | 视觉骨干 | MobileCLIP 图像编码塔 |
| safetensors==0.7.0 | 安全权重格式 | `model.safetensors` 加载 |
| huggingface_hub==0.36.0 | 权重下载 / HF 镜像 | 首启下载走 `HF_ENDPOINT=hf-mirror.com` |
| gradio==6.14.0 | Web 界面框架 | 四页签 GUI 的载体 |
| fastapi==0.136.1 | Web 服务 | gradio 的服务端地基（也直接引用） |
| uvicorn==0.46.0 | ASGI 服务器 | gradio 启动用 |
| opencv_python==4.10.0.84 | 图像读取 / 处理 / 色彩空间 | 推理链核心图像 IO |
| Pillow==12.2.0 | 图像处理 / ICC 色彩配置 | Adobe RGB → sRGB 转换 |
| numpy==2.4.4 | 数值计算 | 全项目基础 |
| PyYAML==6.0.3 | 配置文件解析 | `configs/*.yaml` |
| python_box==7.3.2 | 配置对象化访问 | 原版配置工具（attrdict 风格） |
| attrdict3==2.0.2 | 字典属性化访问 | 原版代码依赖 |
| Requests==2.33.1 | HTTP 请求 | HF 镜像下载 |
| tqdm==4.67.1 | 进度条 | 推理与下载 |
| packaging==26.2 | 版本号比较 | 依赖工具链 |

**可选/仅开发用**（不在精简版，从源码运行也不用于推理）：

| 包 | 用途 | 为什么推理链不需要 |
|---|---|---|
| matplotlib | 画图 | 仅 `data/data_utils.py` 的调试可视化；本仓库已对它做**惰性导入**，缺失时 import 不崩 |
| scikit-image | 直方图均衡 | 同上；`data/data_utils.py` 已做异常兜底 |

## ③ GPU 加速（N 卡，可选，另装一次）

本项目的 GPU 模式只依赖标准 CUDA 版 torch + `device_util.py` 自动分级，不需要以下任何原版重型依赖：

- `bitsandbytes` / `deepspeed` / `flash_attn` / `sglang` / `xformers`

**安装方式**：安装包用户双击 `开启GPU加速.bat`（`packaging/gpu_setup.py`），自动下载 CUDA 版 torch/torchvision wheel 装入嵌入式 runtime：

- 源：清华 TUNA 镜像 `mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cu126`，失败回退 `download.pytorch.org/whl/cu126`
- 版本：与 CPU 版对齐（torch 2.15.0+cu126 / torchvision 0.29.2+cu126），cp312 / win_amd64
- 装法：embedded runtime 无 pip，采用 wheel 即 zip 直接解压进 site-packages（先清 CPU 版防冲突），装后自动跑 `torch.cuda.is_available()` 自检

**源码运行者**：已装 CPU torch 的话，直接换装 CUDA 版即可：

```powershell
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
```

装完后 `app.py` 启动时自动检测 CUDA 并切换 GPU 模式（`device_util.py`），无需其他配置。

## ④ 打包安装包里的差异（面向维护者）

`packaging/requirements-pack.txt` 与 `requirements.txt` 的差别：

1. **剔除 matplotlib / scikit-image**（约省 60–80 MB）：理由见上表；`data/data_utils.py` 已改造为惰性导入 + 文件容错，`tests/test_trim_imports.py` 在"屏蔽这两个包"的模拟环境下验证全链导入通过
2. torch 走 PyPI 默认 CPU 版，安装包运行时无需任何 GPU 依赖
3. 安装包还带嵌入式 Python 3.12（11 MB），**最终 204 MB 安装包 = 嵌入式 Python + 精简后的 66 包 site-packages（0.84 GB 压缩前）+ 应用源码**

## ⑤ 与本机实测一致性

两条 requirements 均已在真实环境完整安装并冒烟验证：

- 开发环境 `vera_venv`（含 matplotlib/skimage）→ 界面 + 推理 + 修图回归全部通过
- 打包环境 `packaging/pack_venv`（不含它们）→ 推理链导入冒烟通过，再经 site-packages 清洗后 staging 冒烟通过
