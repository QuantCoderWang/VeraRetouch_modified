# -*- coding: utf-8 -*-
#
# VeraRetouch GPU 加速一键开启脚本（安装包配套）
#
# 用法：双击安装目录下的「开启GPU加速.bat」即可。
# 行为：
#   1. 用 nvidia-smi 检测 NVIDIA 显卡与驱动支持的 CUDA 版本（无 N 卡则礼貌退出）
#   2. 下载 CUDA 版 torch/torchvision wheel（约 2.6 GB，断点续传）
#      源：清华 TUNA 镜像 -> 失败回退 pytorch 官方源
#   3. 清掉 CPU 版 torch，将 GPU 版解压进 runtime\Lib\site-packages
#   4. 运行自检（torch.cuda.is_available）
# 完成后 launcher/app 启动时 device_util 自动进入 GPU 优先模式（bf16 分级 / 显存自适应分块）。
#
import os
import re
import sys
import shutil
import zipfile
import subprocess

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNTIME = os.path.join(ROOT, "runtime")
PY = os.path.join(RUNTIME, "python.exe")
SP = os.path.join(RUNTIME, "Lib", "site-packages")
CACHE = os.path.join(ROOT, "gpu_cache")          # wheel 下载缓存（断点续传）

# 与安装包内置 CPU torch 版本对齐（防止混装不同版本的行为差异）
PIN_TORCH = "2.15.0"
PIN_TV    = "0.29.2"
CUDA_TAG  = "cu126"
MIN_DRIVER_CUDA = (12, 6)   # cu126 wheel 需要驱动支持 CUDA >= 12.6

INDEX_CANDIDATES = [
    ("tuna",   "https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/{cu}/{{pkg}}/"),
    ("official", "https://download.pytorch.org/whl/{cu}/{{pkg}}/"),
]

import requests


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except Exception:
        return None


def detect_nvidia():
    """返回 (gpu_name, driver_cuda_version tuple) 或 (None, None)。"""
    candidates = ["nvidia-smi",
                  r"C:\Windows\System32\nvidia-smi.exe",
                  r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"]
    r = None
    for cmd in candidates:
        if os.path.exists(cmd) or cmd == candidates[0]:
            r = sh([cmd]) if not os.path.exists(cmd) else sh(cmd)
            if r is not None and r.returncode == 0 and r.stdout:
                break
        r = None
    if not r or r.returncode != 0 or not r.stdout:
        return None, None
    out = r.stdout
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    cuda_ver = (int(m.group(1)), int(m.group(2))) if m else None
    name = None
    first_gpu = re.search(r"^GPU 0[^\n]*\n.*?(Z GPU|([A-Z][\w ]+))", out, re.M)
    names = re.findall(r"\n\s*[A-Za-z][^\n]*\n\s*([A-Z][A-Za-z0-9 .\-]+)\s+\w+", out)
    # 简化：直接抓 NVIDIA 开头的名字
    m2 = re.search(r"NVIDIA[^\n\r]*", out)
    if m2:
        name = m2.group(0).strip()
    return name, cuda_ver


def pick_from_index(text, pkg, pin):
    """从 wheel 目录索引里挑版本：优先 pin 精确版，否则取最高 cp312-win_amd64 版。"""
    pat = re.compile(r"href=\"([^\"]*%2Bcu126-cp312-cp312-win_amd64\.whl|"
                     r"[^\"]*" + re.escape(pkg) + r"[-%+][^\"]*cp312-cp312-win_amd64\.whl)\"",
                     re.I)
    hrefs = re.findall(r"href=\"([^\"]+)\"", text)
    best = None
    best_ver = None
    pin_found = None
    for h in hrefs:
        base = os.path.basename(h)
        m = re.match(re.escape(pkg) + r"-(\d+\.\d+\.\d+)(%2B|\+)cu126-cp312-cp312-win_amd64\.whl",
                     base, re.I)
        if not m:
            continue
        ver = tuple(int(x) for x in m.group(1).split("."))
        if pin_found is None and m.group(1) == pin:
            pin_found = h
        if best_ver is None or ver > best_ver:
            best, best_ver = h, ver
    return pin_found or best


def fetch_index(pkg):
    for src_name, tpl_src in INDEX_CANDIDATES:
        url = tpl_src.format(pkg=pkg)
        url = url.replace("{cu}", CUDA_TAG)
        try:
            rr = requests.get(url, timeout=20)
            if rr.status_code == 200 and ".whl" in rr.text:
                print(f"[gpu] 索引可用：{src_name}（{pkg}）", flush=True)
                return url, rr.text
        except Exception:
            pass
    return None, None


def download(url, dest):
    """断点续传下载（Range 支持的 HTTP 源一致适用）。"""
    part = dest + ".part"
    done = os.path.getsize(dest) if os.path.exists(dest) else 0
    headers = {}
    if os.path.exists(part):
        have = os.path.getsize(part)
        if have > 0:
            headers["Range"] = f"bytes={have}-"
    while True:
        # .part 续传 或 完整文件重新校验
        have = os.path.getsize(part) if os.path.exists(part) else 0
        with requests.get(url, stream=True, timeout=60, headers=headers) as resp:
            total = resp.headers.get("Content-Length")
            total = int(total) + have if resp.status_code == 206 and total else int(total or 0)
            mode = "ab" if resp.status_code == 206 else "wb"
            last_pct = -1
            with open(part, mode) as f:
                for chunk in resp.iter_content(1024 * 512):
                    f.write(chunk)
                    if total:
                        pct = int(f.tell() * 100 / total)
                        if pct != last_pct and pct % 2 == 0:
                            print(f"\r      {pct}%  ({f.tell()/1024/1024:.0f}/{total/1024/1024:.0f} MB)",
                                  end="", flush=True)
                            last_pct = pct
        print(flush=True)
        if total and os.path.getsize(part) >= total:
            os.replace(part, dest)
            return dest
        print("[gpu] 下游中断，5 秒后从断点续传…", flush=True)
        import time as _t
        _t.sleep(5)


def main():
    print("=" * 56)
    print("VeraRetouch GPU 加速开启工具")
    print("=" * 56, flush=True)

    name, drv_cuda = detect_nvidia()
    if name is None:
        print("[gpu] 未检测到 NVIDIA 显卡（nvidia-smi 不可用或无 N 卡设备）。")
        print("[gpu] 本机将继续以 CPU 模式运行，无需此工具。按回车退出…")
        try:
            input()
        except EOFError:
            pass
        return 0
    print(f"[gpu] 检测到显卡：{name}")
    if drv_cuda is None or drv_cuda < MIN_DRIVER_CUDA:
        print(f"[gpu] 驱动支持的 CUDA 版本不符合要求（需 >=12.6，检测到 {drv_cuda}）。")
        print("[gpu] 请到 nvidia 官网更新显卡驱动后重试；当前继续用 CPU。按回车退出…")
        try:
            input()
        except EOFError:
            pass
        return 1
    print(f"[gpu] 驱动 CUDA 版本：{drv_cuda[0]}.{drv_cuda[1]}（满足 {CUDA_TAG} wheel 要求）", flush=True)

    # 已装 CUDA torch 的快捷检查
    probe = sh([PY, "-c", "import torch; print(torch.__version__, torch.cuda.is_available())"])
    if probe and probe.returncode == 0:
        ver, ok = probe.stdout.strip().rsplit(" ", 1) if " " in probe.stdout.strip() else ("", "False")
        if ok == "True" and "cu1" in ver:
            print(f"[gpu] GPU 计算库已就绪（{ver}），无需重复安装。")
            # 依然打印设备信息以确认
            info = subprocess.run([PY, "-c", "from device_util import resolve_device_and_dtype; "
                                               "import torch; d,t=resolve_device_and_dtype(); "
                                               "print(torch.cuda.get_device_name(0))"],
                                  capture_output=True, text=True, cwd=ROOT)
            if info.returncode == 0:
                print(f"[gpu] {info.stdout.strip()}")
            return 0

    os.makedirs(CACHE, exist_ok=True)
    wheels = {}
    for pkg, pin in (("torch", PIN_TORCH), ("torchvision", PIN_TV)):
        base_url, text = fetch_index(pkg)
        if not text:
            print(f"[gpu] [失败] 无法访问任何 {pkg} wheel 索引（镜像与官方源都不通）。", flush=True)
            return 1
        href = pick_from_index(text, pkg, pin)
        if not href:
            print(f"[gpu] [失败] 索引里没找到 {CUDA_TAG}/cp312/win_amd64 的 {pkg} wheel。", flush=True)
            return 1
        wheel_url = href if href.startswith("http") else base_url.rsplit("/", 1)[0] + "/" + href
        wheels[pkg] = (wheel_url,
                       os.path.join(CACHE, os.path.basename(href.replace("%2B", "+"))))
        print(f"[gpu] 将下载 {os.path.basename(href)}", flush=True)

    for pkg, (wurl, wpath) in wheels.items():
        print(f"[gpu] 下载 {pkg}（{wurl}）", flush=True)
        try:
            download(wurl, wpath)
        except Exception as e:
            print(f"\n[gpu] [失败] {pkg} 下载失败：{e}\n（已支持断点续传，重新运行本工具将继续下载）", flush=True)
            return 1

    # 安装：清 CPU 版再解压 GPU 版（wheel = zip；嵌入式 runtime 无 pip）
    pkg = "torch"
    try:
        for d in ("torch", "torchvision"):
            target = os.path.join(SP, d)
            if os.path.isdir(target):
                print(f"[gpu] 清理旧版 {d} …", flush=True)
                shutil.rmtree(target, ignore_errors=True)
        for wpath in (wheels["torch"][1], wheels["torchvision"][1]):
            print(f"[gpu] 安装 {os.path.basename(wpath)} …", flush=True)
            with zipfile.ZipFile(wpath) as zf:
                zf.extractall(SP)
    except Exception as e:
        print(f"[gpu] [失败] 安装出错：{e}", flush=True)
        return 1

    # 自检
    check = subprocess.run(
        [PY, "-c",
         "import torch, torchvision; ok = torch.cuda.is_available(); "
         "print(torch.__version__, torchvision.__version__, ok); "
         "print(torch.cuda.get_device_name(0)) if ok else None"],
        capture_output=True, text=True, timeout=300, cwd=ROOT)
    out = (check.stdout or "") + (check.stderr or "")
    if check.returncode == 0 and "True" in (check.stdout or ""):
        print("-" * 56)
        print("[gpu] ✔ GPU 加速开启成功！下次启动 VeraRetouch 即自动进入 GPU 模式")
        if check.stdout:
            print("      " + check.stdout.strip().replace("\n", "\n      "))
        print("-" * 56, flush=True)
        return 0
    print(f"[gpu] [失败] 安装后自检未通过：\n{out}\n按回车退出…")
    try:
        input()
    except EOFError:
        pass
    return 1


if __name__ == "__main__":
    sys.exit(main())
