<div align="center">

# VeraRetouch GUI —— 全中文图形修图界面

**基于 VeraRetouch 的二次开发：CPU 即装即用 / N 卡一键加速 / Windows 安装包免装 Python / 38 参数滑块式图形输入**

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU%20%7C%20CUDA-orange)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-6.x-green)](https://gradio.app/)

**原项目：[OpenVeraTeam/VeraRetouch](https://github.com/OpenVeraTeam/VeraRetouch)　·　论文：[arXiv:2604.27375](https://arxiv.org/abs/2604.27375)**

*本仓库是在上述原项目基础上的工程化二次开发（中文界面 / 交互改造 / Windows 安装包）；模型与算法、权重全部出自原作者。详情见下方致谢与许可证章节。*

</div>

---

## 🤖 AI 编程声明（重要）

**本项目是人类设计 + AI 编程协作完成的产物**，特此如实声明：

| 角色 | 职责 |
|---|---|
| **人类开发者（项目主要负责人）** | 提供修改思路与全部设计决策：交互形态（Gradio 五页签）、38 参数图形滑块交互（基于实测反馈去掉早期 JSON 专家模式）、CPU 移植 / GPU 分级策略、参考图必填等校验规则、打包架构（嵌入式 Python + Inno Setup）、汉化文案与体验取舍、全部功能性 bug 的定位与验收 |
| **生成式 AI 编程助手** | 在人类给出的设计思路与决策下辅助编写具体代码、构建脚本与文档 |

人类开发者对 AI 生成的全部代码**逐一审阅、实测验证**（界面冒烟测试、回归测试、双形态安装包真机安装验证）。AI 仅是工具，本项目的软件质量与设计责任由人类开发者承担。

> 依据 Apache-2.0 与主流开源社区的惯例，AI 辅助生成的衍生作品可正常以开源许可证发布；本声明意在向使用者透明说明开发过程。

---

## 📖 项目简介

[VeraRetouch](https://github.com/OpenVeraTeam/VeraRetouch)（浙大 & vivo BlueImage，论文见顶部链接）是一个轻量级、完全可微的多任务照片修图框架。本仓库在其**推理代码**基础上做了大幅二次开发，使其成为人人可用的桌面工具：

1. **全面汉化**：界面、按钮、进度与错误提示全中文（`ui/i18n.py` 集中管理）。
2. **从"手写 JSON 文件喂命令行"变为"人人都会用"**：参数修图页签将配色 / 细节等 **38 个参数**全部做成**滑杆 + 数值框**，每个参数附中文作用说明；滑块调到哪模型立刻懂（内部自动生成参数文件驱动管线，用户全程无需接触 JSON）。
   > 早期版本曾提供"图形输入 ↔ JSON 专家模式"双模式切换；经用户实测反馈后已**移除 JSON 输入模式，只保留图形（滑块）输入**。
3. **仅 CPU 的电脑完全可以运行**：原版面向 GPU，本仓库补齐 CPU 路径（`CPU_MIGRATION_AND_UI_PLAN.md` 有完整移植记录），无独显用户开箱即用。
4. **NVIDIA 显卡一键加速**：安装包默认内置 CPU 计算、开箱即用；想跑满 N 卡，双击一次安装目录里的 `开启GPU加速.bat`（自动检测显卡并下载 CUDA 计算库）即可——**之后每次启动都自动进入 GPU 模式**，bf16 精度按算力自动分级、渲染分块按显存自适应，零手动配置。
5. **Windows 一键安装包**（`installer/`）：**无需安装任何 Python**，双击即用；输出图片自动按会话存档，导出格式可选（默认跟随输入格式），启动后浏览器自动打开修图界面。

---

## ✨ 功能亮点

| 功能 | 说明 |
|---|---|
| 🖼 一键自动修图 | 上传照片 → 自动推理出最佳参数 → 直接出图 |
| 📥 支持格式 | 输入常规图片格式：jpg / jpeg / png / webp / bmp / tif / tiff（上传框提示"支持 jpg / png"）；相机 RAW（CR2 / NEF / DNG 等）暂不支持 |
| 🎨 风格修图 | 输入风格描述（如"梦幻明亮的粉色调"）一键套用 |
| 🎚 参数修图 | 38 参数滑块 / 数值框精修，附中文作用说明；一键全部归零 / 载入示例参数 |
| 💾 导出格式可选 | 默认**跟随输入格式**（jpg 进 jpg 出）；也可固定选 PNG（无损）/ JPG / WebP，结果预览与下载都是实际格式，不再被网页转成 webp |
| 🔍 参考图修图 | 学习参考图的修图风格并迁移到你的照片（修图前 + 修图后两张参考图必填，缺一张即弹窗报错、不允许执行） |
| 🗂 批量自动修图 | 选择待修图文件夹 + 输出文件夹即全自动逐张修图：输出跟随输入格式、每张生成同名分析记录 txt、汇总 `batch_summary.txt`、完成有系统提醒（提示音 + 通知条）；中断后重跑自动续 |
| 🧹 画质保障 | 每个页签都有「输出降噪」下拉框（**默认关闭**）：关闭=模型裸输出零改动；轻/中/强为"细节回贴式"降噪（先强降噪、再把原图高频细节按比例贴回），抑制提亮暗部时被放大的传感器噪点；输出自动按照片 EXIF 方向转正，不再横竖颠倒 |
| 💻 无显卡友好 | 无需显卡开箱即用（CPU 模式），核显笔记本 / 老电脑可跑 |
| 🚀 N 卡极速模式 | 运行一次 `开启GPU加速.bat` 自动装 CUDA 计算库（断点续传），此后每次启动自动 GPU 加速 |
| ☁️ 首启权重下载 | 在线版 ~204 MB，首次启动自动从 hf-mirror 拉权重（约 2.3 GB，断点续传） |
| 📦 全量离线版 | ~1.44 GB 权重内置，无网络环境安装即用 |

> 五个页签：**一键自动修图 / 风格修图 / 参数修图 / 参考图修图 / 批量自动修图**

## 📚 文档导航

| 文档 | 内容 |
|---|---|
| [docs/CHANGES_VS_UPSTREAM.md](docs/CHANGES_VS_UPSTREAM.md) | **本项目所有改进明细**（CPU 可运行方案 / 滑块界面 / 汉化 / 打包 / 修复的 bug） |
| [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md) | **依赖包说明**：每个包的用途、GPU 专用可选项、打包精简了什么 |
| [docs/README_upstream.md](docs/README_upstream.md) | 原版 VeraRetouch README 存档（论文、作者、原功能） |
| [CPU_MIGRATION_AND_UI_PLAN.md](CPU_MIGRATION_AND_UI_PLAN.md) | CPU 移植与界面改造全过程工程记录 |
| [批量自动修复_技术方案.md](批量自动修复_技术方案.md) | 批量执行自动修复放记录 |
| [界面汉化_打包安装_参数滑块_改造方案.md](界面汉化_打包安装_参数滑块_改造方案.md) | 应用首版改造方案 |

---

## 🚀 快速开始

### 方式一：Windows 安装包（推荐，无需 Python 环境）

因为文件太多，未置于release页面中，需要开发者手动构建，下面仅提供两种构架思路：

| 形态 | 文件 | 体积 | 适用 |
|---|---|---|---|
| 在线版 | `VeraRetouch_Setup.exe` | ~204 MB | 能联网，首启自动下载权重（约 2.3 GB） |
| 离线全量版 | `VeraRetouch_Full_Setup.exe` | ~1.44 GB | 无法联网或网速慢，装完即用 |

双击 exe 安装 → 桌面双击 **VeraRetouch** 图标 → 等待模型加载 → 浏览器自动打开修图界面。**关闭控制台窗口即退出程序**。

有 N 卡想加速：安装后双击安装目录里的 **`开启GPU加速.bat`** 一次即可（详见下方硬件说明）。

### 方式二：从源码运行（开发者）

```powershell
git clone https://gitee.com/<你的用户名>/veraretouch-gui.git
cd veraretouch-gui
python -m venv vera_venv
vera_venv\Scripts\pip install -r requirements.txt
# 权重：从 hf-mirror 下载（首次约 2.3 GB，断点续传）
python packaging\download_weights.py
# 启动
vera_venv\Scripts\python app.py   # 模型加载完成后自动打开默认浏览器
```

浏览器没有自动弹出时，手动访问 `http://127.0.0.1:7860`。

有 NVIDIA 显卡并想用 GPU：`pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cu126 --force-reinstall`（详见 [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md)）。

#### 手动放置权重（可选）

```
checkpoints/
├─ VeraRetouch/
│  └─ model.safetensors   (~2.2 GB，来自 https://huggingface.co/Gyh68/VeraRetouch)
└─ encoder_renderer.pth   (~180 MB，来自 https://huggingface.co/Gyh68/VeraRetouch.Encoder_Renderer)
```

---

## 🖥 硬件与运行要求

| 设备 | 行为 |
|---|---|
| 无独显 / 核显 / 老电脑 | **开箱即用**（CPU 模式，fp32，速度较慢但不影响功能；内存建议 ≥ 8 GB） |
| NVIDIA 显卡 | 先运行一次安装目录里的 `开启GPU加速.bat`（自动下载 CUDA 计算库约 2.6 GB，断点续传；显卡驱动需支持 CUDA ≥ 12.6），完成后**每次启动自动 GPU 加速** |
| 精度与显存策略 | 算力 ≥ 8.0（RTX 30 系起）自动 bf16；显存 ≥ 12 GB 时渲染不分块；均自动判断，无需配置 |

## 📁 项目结构

```
├── app.py                    # 主界面（Gradio 五页签；38 参数滑块 + 批量模式）
├── inference.py              # 推理入口（CLI）
├── utils.py / device_util.py # 渲染器获取 / 设备-精度-分块自适应
├── ui/                       # 界面文案与参数定义（i18n.py / params_def.py）
├── model/ / llava/ / data/   # 模型实现与数据管线
├── ml-veraretouch/           # 原版自带的 iOS/macOS MLX 部署工程（随上游保留）
├── configs/                  # 模型与推理配置
├── checkpoints/              # 权重目录（不入库，首次运行下载）
├── data_samples/             # 示例图片与示例参数
├── packaging/                # 打包与分发
│   ├── build_installer.ps1   #   一键打 Windows 安装器
│   ├── download_weights.py   #   hf-mirror 权重下载（断点续传）
│   ├── gpu_setup.py          #   N 卡 CUDA 计算库自动下载安装
│   ├── 开启GPU加速.bat        #   安装目录里的一键 GPU 入口（调 gpu_setup.py）
│   └── launcher.py           #   安装版启动器（端口就绪才开浏览器）
├── installer/veraretouch.iss # Inno Setup 脚本（/DFULL 切换全量版）
├── tests/                    # 秒级回归冒烟测试（不加载权重）
├── docs/                     # 改进明细 / 依赖说明 / 原版 README 存档
└── CPU_MIGRATION_AND_UI_PLAN.md  # CPU 移植与界面改造全过程记录
```

## 🧪 维护者区：回归测试与打包

改动界面 / 打包相关代码后先跑冒烟测试（秒级，不加载权重）：

```powershell
vera_venv\Scripts\python tests\ui_smoke_test.py       # 界面构建：38 滑块 / 10 按钮 / 9 下拉框（导出格式+输出降噪）/ 5 页签 / 批量端到端
vera_venv\Scripts\python tests\test_trim_imports.py   # 精简依赖环境导入链
```

重新打安装包：

```powershell
powershell packaging\build_installer.ps1 -ReuseVenv    # 形态 1 在线版（~204 MB）
iscc /DFULL installer\veraretouch.iss                  # 形态 3 全量离线版（~1.44 GB）
```

---

## 🙏 致谢与出处

本项目**建立在原作团队的工作之上**，所有核心模型与算法完全属于原项目：

* **论文**：VeraRetouch: A Lightweight Fully Differentiable Framework for Multi-Task Reasoning Photo Retouching ([arXiv:2604.27375](https://arxiv.org/abs/2604.27375))
* **原项目（含 macOS/iOS 部署）**：[OpenVeraTeam/VeraRetouch](https://github.com/OpenVeraTeam/VeraRetouch)
* **作者**：Yihong Guo, Youwei Lyu, Jiajun Tang, Yizhuo Zhou, Hongliang Wang, Jinwei Chen, Changqing Zou, Qingnan Fan —— 浙江大学、vivo BlueImage Lab、中国科学院大学
* **项目主页**：https://apollo-yi.github.io/VeraRetouch/
* 本仓库在其基础上完成：CPU 移植、Gradio 全中文界面、38 参数滑块交互、参考图校验强化、导出格式可选、批量自动修图、输出降噪、Windows 打包分发、GPU 一键加速（明细见 [docs/CHANGES_VS_UPSTREAM.md](docs/CHANGES_VS_UPSTREAM.md)），并附 `NOTICE` 著作权声明。

**模型权重版权归原项目所有**，使用限制随权重组件一并生效。

---

## 📜 许可证（License）

本仓库继承上游双许可结构（原版 `LICENSE` 未做修改）：

| 组成 | 许可 |
|---|---|
| **代码**（本仓库全部源码，含二开部分） | **Apache License 2.0**（随附 `NOTICE` 著作权声明） |
| **模型权重**（checkpoints 及其衍生模型） | **仅限学术研究，禁止商业用途**（原版作者声明） |

