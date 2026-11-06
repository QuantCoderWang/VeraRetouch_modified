#
# VeraRetouch launcher for the installed (portable) distribution.
# Console window doubles as the "close to exit" switch (children die with the
# attached console). Per 改造方案 §4.6: check weights -> start app.py -> poll
# port -> open browser once the server is actually ready.
#
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))          # install root
APP = os.path.join(ROOT, "app")
PY = os.path.join(ROOT, "runtime", "python.exe")
if not os.path.exists(PY):  # dev fallback: run from repo with the dev venv
    APP = ROOT
    PY = sys.executable
PORT = 7860
URL = f"http://127.0.0.1:{PORT}"

# 嵌入式 Python 的 _pth 模式会把 launcher.py 所在目录排除在 sys.path 之外，
# 必须显式加入，否则 from download_weights import ... 直接 ModuleNotFoundError
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.chdir(APP)  # app.py resolves ./checkpoints, ./configs, ./data_samples relative to cwd
sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None


def wait_for_server(deadline_sec=900):
    t0 = time.time()
    while time.time() - t0 < deadline_sec:
        if proc.poll() is not None:  # app.py died
            return False
        try:
            with urllib.request.urlopen(f"{URL}/info", timeout=2):
                return True
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(1.0)
    return False


print("=" * 56)
print("VeraRetouch 修图工具")
print("首次启动需要下载模型权重并加载模型，请稍候…")
print("=" * 56, flush=True)

# 1) weights (first run only; resumable)
try:
    from download_weights import ensure_weights
    ensure_weights(APP)
    print("\n")
except ModuleNotFoundError:
    print("[launcher] download_weights.py 未随包安装，跳过权重检查")
except Exception as e:
    print(f"[launcher] 权重下载失败：{e}\n程序即将退出；稍后可重新双击图标（支持断点续传）")
    try:
        input("按回车关闭窗口…")
    except EOFError:
        pass
    sys.exit(1)

# 2) start the app
# --no-inbrowser: app.py 自带 inbrowser（服务就绪即开浏览器），但安装版由本启动器
# 统一负责：端口探活（/info 可访问）成功后再开，与控制台日志时序一致，避免开两次。
proc = subprocess.Popen([PY, "app.py", "--no-inbrowser"], cwd=APP)
print(f"[launcher] 应用启动中：{PY} app.py", flush=True)

# 3) wait for the HTTP server, THEN open the browser (port probe per §4.6)
if wait_for_server():
    print(f"[launcher] 服务已就绪，打开浏览器：{URL}")
    webbrowser.open(URL)
else:
    print("[launcher] 等待服务超时或应用已退出，请查看上方日志", flush=True)
    try:
        input("按回车关闭窗口…")
    except EOFError:
        pass
    sys.exit(1)

# 4) the console window is the exit switch: closing it ends the app
try:
    proc.wait()
except KeyboardInterrupt:
    proc.terminate()
if proc.returncode not in (0, None):
    print(f"[launcher] 应用异常退出（代码 {proc.returncode}），请查阅上方日志")
    try:
        input("按回车关闭窗口…")
    except EOFError:
        pass
