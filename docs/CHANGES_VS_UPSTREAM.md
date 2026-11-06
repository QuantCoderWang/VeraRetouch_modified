# 本项目相对于 VeraRetouch 原版的改进（CHANGES）

原版 VeraRetouch 面向**研究者**：GPU 环境 + 命令行 + 手写 JSON 参数文件。本项目把它改造成**普通用户可用**的桌面工具。以下按模块列出全部实质性改进，供审查与回归对照。

> 原著作：VeraRetouch: A Lightweight Fully Differentiable Framework for Multi-Task Reasoning Photo Retouching（arXiv:2604.27375，OpenVeraTeam）
> 本仓库所有改动均为在其推理代码基础上的**工程化二次开发**，不改动模型核心算法；模型权重与原论文一致。

---

## 1. 仅 CPU 的电脑完全可以运行 ⭐

| 项 | 说明 |
|---|---|
| 适用机器 | 无独显的老电脑 / 核显本本 / 办公机 |
| 参考配置 | Windows 10/11 + 双核以上 CPU + 8 GB 内存（模型 fp32 有 ~2.2 GB，内存建议 ≥8 GB）即可运行 |
| 存储 | 程序 ~250 MB + 权重 ~2.4 GB |
| 实现方式 | `CPU_MIGRATION_AND_UI_PLAN.md` 完整记录的 CPU 移植：device 分流、dtype 约束、渲染器 CPU 路径验证 |
| 体验 | 无 GPU 也能出图，速度相对慢；有 N 卡则可加速 |

**有 NVIDIA 显卡的加速路径（一次性）**：安装包默认附 CPU 版 torch，保证所有机器开箱能用；GPU 机器双击安装目录的 **`开启GPU加速.bat`**（`packaging/gpu_setup.py`）即可——自动检测显卡与驱动 CUDA 版本（需 ≥12.6），从清华 TUNA 镜像（失败回退官方源，断点续传）下载 CUDA 版 torch/torchvision（约 2.6 GB），以 wheel=zip 方式装入嵌入式 runtime，再运行 `torch.cuda.is_available()` 自检。

**装好之后每次启动全自动**：`device_util.py` 按算力自动选精度（bf16 需算力 ≥ 8.0），按显存自适应渲染分块（≥12 GB 不分块），CPU-only 机器始终自动回退 CPU——零手动配置。

## 2. 图形界面：JSON 输入文件 → 人人可用的滑块交互 ⭐

原版修图参数要手写 JSON 文件喂给 CLI。本仓库在 `app.py` 重写为 Gradio 界面：

- **参数页签 38 个参数全部做成滑杆 + 数值框**组合，每个参数带**中文作用说明**（`ui/params_def.py` 参数定义表：中文名 / 范围 / 步长 / 默认值 / 说明文案 / 布局序号）
- 一键**全部归零**、**载入示例参数**；38 滑块提交前有防错位断言（组件顺序 ↔ 布局键序）
- **启动即用：服务就绪后自动打开默认浏览器**（`app.py` 的 `--inbrowser/--no-inbrowser`，默认开启）；安装版启动器 `launcher.py` 采用更稳的端口探活后开浏览器，并给 app.py 传 `--no-inbrowser` 避免重复弹出
- **导出格式可选（2026-11-05 用户反馈）**：gradio 6 的 `gr.Image` 默认 `format="webp"`，返回 PIL 对象会被前端静默重编码为 webp——**根因**。现改为各处理器自行按选定格式落盘（`save_result(fmt_key="auto"/"png"/"jpg"/"webp"`；JPG/WebP 质量 95，JPG 自动转 RGB 防 alpha 通道报错），并把**实际文件路径**返回给 `gr.Image`（Gradio 对文件路径原样展示，见 `gradio/image_utils.py` 的 `save_image`）；四个页签各加"导出格式"下拉框，默认**跟随输入格式**（输入 jpg 出 jpg；识别不了的扩展名回退 PNG）。CLI 侧未动（`inference.py` / `infer_ref_retouch.py` 本就落盘无损 PNG，无此问题）
- **输出画质排查与降噪/EXIF 修复（2026-11-05/06）**：用户反馈"输出图像质量下降太多"。用真实输入/输出对量化排查（`tests/_quality_probe*.py` 留档）证实：保存 q95 重编码损失 53.6 dB（肉眼不可见）、分辨率/锐度/渲染管线全部无损、无 chunk 接缝伪影——**真正原因是模型对暗光照片大幅提亮暗部时，输入端传感器噪声随亮度曲线被同倍率放大**（亮度高频噪声 5.9 级、色度仅 0.5 级，上游 CLI 存无损 PNG 亦同）。
  - **输出降噪（方案一·细节回贴，2026-11-06 按用户二测反馈定稿）**：用户测试裸 NLM 后反馈"细节损失也非常大、体积大幅下降"，故改为细节回贴式：`final = dn + α×(orig−dn)`，五页签统一「输出降噪」下拉框（关闭/轻/中/强，**默认关闭**——模型裸输出零改动）。
  - **EXIF 方向修复**：PIL 保存不写 EXIF，摄影方向标签为 6/8 的照片输出后横竖颠倒（实测 11 张中 1 张）——`save_result` 现按源图 EXIF 方向把输出像素转正，不再依赖标签。
- **第五页签「批量自动修图」（2026-11-05，方案文档见 `docs/批量自动修图_技术方案.md`）**：选择待修图文件夹 + 输出文件夹（原生 Windows 文件夹对话框，经 PowerShell `FolderBrowserDialog`，规避嵌入式 runtime 无 tkinter 的限制）→ 全自动逐张 auto 修图。每张输出**跟随输入格式**的结果图 + **同名 .txt**（模型分析与修图计划原文）+ 全批 `batch_summary.txt` 汇总；单张失败记入汇总继续、显存 OOM 先清缓存重试一次；每 10 张 `gc.collect()` 低频保险丝（防长批次内存侵蚀，§4.9）；中断重跑自动跳过已完成（断点续跑）；双进度条（总体=时间加权+动态 ETA、单图=既有流式逻辑）+ 完成提醒（通知条 + Windows 提示音）
- **参考图修图严格必填校验（2026-11-05 用户要求）**：输入图、参考图·修图前、参考图·修图后三张缺一即**弹窗报错、不允许执行**（Gradio `gr.Error` 模态提示）；移除了原实现"修图前留空则用输入图顶替"的隐性 fallback（该行为会让无参考图的运行产出无意义结果）

- **相机 RAW 格式：曾按用户需求实现，后按用户决定整体回退（2026-11-06 当日三轮）**：现**不支持**相机 RAW（CR2/CR3/NEF/ARW/DNG 等），仅支持常规格式 jpg / jpeg / png / webp / bmp / tif / tiff（上传提示文案为"支持 jpg / png"）。回退范围：`RAW_EXTS`/`prepare_input`/`_is_raw` 解码层、`batch_loop`/`_run_retouch`/`ref_handler` 接入、`scan_handler` RAW 计数、i18n RAW 文案、`tests/ui_smoke_test.py::test_raw_support`、gradio 前端 `accepted_types` 补丁（五处 site-packages 已恢复 `image/*`）、`packaging/patch_gradio_raw_upload.py` 及其 `build_installer.ps1` 步骤。**保留项**：`app.py: _patch_gradio_asset_no_cache()`（gradio `/assets/*` 响应补 `Cache-Control: no-cache` + 304 条件请求——该版本 gradio 资产响应不带缓存控制头，浏览器按启发式缓存数天导致前端更新不生效；与 RAW 无关，对所有界面更新有益）。回退原因：用户实测大疆运动相机 DNG 不在 LibRaw 支持表内，rawpy 无法解码，靠机内嵌预览兜底画质有折损，用户决定退回仅常规格式。留档：`tests/_check_rawpy.py`、`tests/_sample_raw/`（Canon SRAW 样张）、`tests/_revert_gradio_asset_patch.py`（资产补丁回退脚本）。

- **JSON 专家模式已按用户反馈移除（2026-11-05）**：目前**唯一输入方式就是图形滑块**。滑块值在提交时内部转成 `param.json` 驱动推理管线（`sliders_to_json`），用户全程不接触 JSON。早期版本的双模式同步/解析容错逻辑随该模式一并移除（相关回归测试 `tests/repro_invalid_json_switch.py` 删除，滑块往返转换回归保留在 `tests/ui_smoke_test.py`）

## 3. 全中文界面

- 页签 / 按钮 / 提示 / 进度 / 报错全部中文，文案集中在 `ui/i18n.py`（改文案不用翻界面代码）
- 敏感路径（bug 反馈、安装说明）随手可得

## 4. Windows 一键安装包（免装 Python）⭐

| 特性 | 说明 |
|---|---|
| 免 Python 环境 | 嵌入式 Python 3.12.10（`runtime/`）随包分发，**用户机器不需要装任何东西** |
| 精简打包 | site-packages 从 venv 原始态清洗到 0.84 GB（删 C 头文件 9000+、HTML 模板白名单保护、dist-info 精简），最终安装包 204 MB |
| 两种形态 | 形态 1 在线版（~204 MB，首启下载权重）+ 形态 3 全量版（~1.44 GB，权重内置），同一 iss `/DFULL` 开关产出 |
| 首启下载 | `packaging/download_weights.py`：hf-mirror 镜像、断点续传、下载完尺寸校验 |
| 端口就绪才弹浏览器 | `packaging/launcher.py` 轮询 `127.0.0.1:7860/info`，服务真正起来才开浏览器（避免白页） |
| 每用户安装 | `{localappdata}` 安装，无需管理员权限/无 UAC 弹窗，中文安装向导 |

## 5. 相对原版修复的 bug / 健壮性（与逐条对照）

| # | 问题 | 修复 |
|---|---|---|
| 1 | `data/data_utils.py` 模块级 `Image.open` 引用仓库中不存在的文件 → anyone import 即崩 | 改为 try/except 容错，`ADOBE_PROFILE` 缺失时仅在使用处报错 |
| 2 | `data/data_utils.py` 模块级 matplotlib/skimage 导入 → 无这俩包的环境直接不可用 | 惰性导入（推理链不依赖它们） |
| 3 | 模式切换时 JSON 解析失败会阻断视觉切换 | 处理器内降级为 Warning，保留滑块原值（该模式已随 JSON 专家模式整体移除，问题不复存在） |
| 4 | 38 滑块组件与协议键序不一致的错位隐患 | `layout_order_keys()` 显式映射 + 构建期逐键断言（`tests/ui_smoke_test.py`） |
| 5 | GPU 机上 dtype / OOM 风险 | `device_util.py` 统一分级管理（CPU / CUDA 计算能力检测 / 显存自适应 chunk） |
| 6 | `app.py` / `inference.py` 把生成参数 `--max_new_tokens`（4096）误传给 `AutoTokenizer.from_pretrained` 的 `model_max_length`，覆盖了训练方保存在 `tokenizer_config.json` 里的权威值（8192），并被 `infer_dataset.py` 当作**输入侧编码截断上限**使用，过长输入会被错误截短 | 新增独立的 `--model_max_length` CLI 参数（默认 None 不传 kwarg，让 tokenizer 配置文件原值 8192 生效）；与 `--max_new_tokens`（生成长度）彻底解耦（2026-11-05） |

## 6. 全新工程化资产（原版没有）

- `packaging/`：构建脚本（每步冒烟防删除过头）、启动器、权重下载器、精简 requirements
- `installer/veraretouch.iss`：双形态 Inno Setup 安装脚本（`/DFULL` 切换）
- `tests/`：2 份不加载权重的秒级回归测试（`ui_smoke_test.py`：界面构建断言 38 滑块 / 10 按钮 / 9 下拉框（4 导出格式 + 5 输出降噪）/ 5 页签 + 参数往返 + 导出格式选择 + 输出降噪与 EXIF 转正 + 批量辅助函数/端到端 + 相机 RAW 支持；`test_trim_imports.py`：精简依赖导入链）
- 文档：CPU 移植全过程记录（`CPU_MIGRATION_AND_UI_PLAN.md`）、改造方案台账（`界面汉化_打包安装_参数滑块_改造方案.md`）、本依赖说明（`docs/DEPENDENCIES.md`）、AI 编程声明（README 置顶）

## 7. 开发过程声明

本项目由 **人类开发者在积极的修改思路与设计决策驱动下、由 AI 编程助手在人类审阅与实测验证下共同完成**，详见 README 顶部《AI 编程声明》。
