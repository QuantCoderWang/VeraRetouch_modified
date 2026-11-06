# =====================================================================
# VeraRetouch 打包总脚本（改造方案 §4.8 全流程）
# 产物：installer\dist\VeraRetouch_Setup.exe（目标 < 700 MB，形态 1 无权重）
#
# 前置条件：
#   1. 开发机装有 Python 3.12（用于干净安装依赖；安装包内自带嵌入式 Python，
#      用户机无需任何 Python）
#   2. Inno Setup 6+（iscc.exe 在 PATH，或用 -Iscc 指定完整路径）
#   3. 可访问 pip 镜像（默认阿里云）
#
# 用法（PowerShell，项目根目录）：
#   .\packaging\build_installer.ps1
#   .\packaging\build_installer.ps1 -SkipBuildStaging        # 复用已有 staging 仅重编安装器
#   .\packaging\build_installer.ps1 -Python "D:\python3.12\python.exe"
# =====================================================================
[CmdletBinding()]
param(
    [string]$Python          = "D:\python3.12\python.exe",
    [string]$Iscc            = "iscc",
    [switch]$SkipBuildStaging,
    [switch]$ReuseVenv,                                                        # 复用已装好依赖的 pack_venv（清洗规则修正后重跑用）
    [string]$EmbedPyUrl      = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip",
    [string]$EmbedPyUrlCn    = "https://mirrors.huaweicloud.com/python/3.12.10/python-3.12.10-embed-amd64.zip",
    [string]$PipIndex        = "https://mirrors.aliyun.com/pypi/simple/"
)

$ErrorActionPreference = "Stop"
$Root  = Split-Path -Parent $PSScriptRoot
$Stage = Join-Path $Root "packaging\staging"
$Logs  = Join-Path $Root "packaging\logs"
New-Item -ItemType Directory -Force -Path $Logs | Out-Null
Push-Location $Root   # 导入冒烟依赖项目根目录（sys.path[0]=cwd）
try {

function Step($msg) { Write-Host ("==> " + $msg) -ForegroundColor Cyan }
function Info($msg) { Write-Host ("    " + $msg) }
function Smoke($py, $tag) {
    # 每步精简后的导入冒烟（§4.3：防删过头）
    & $py -c "import torch, cv2, transformers, timm, diffusers, gradio, yaml; import inference, utils" 2>&1 |
        Select-Object -Last 5 | ForEach-Object { Info $_ }
    if ($LASTEXITCODE -ne 0) { throw "冒烟测试失败（$tag）：精简过头了，回退最后删除的内容" }
    Info "冒烟通过（$tag）"
}

# ---------------------------------------------------------------- 阶段①–⑤
if (-not $SkipBuildStaging) {
    if (Test-Path $Stage) { Remove-Item -Recurse -Force $Stage }
    New-Item -ItemType Directory -Force -Path (Join-Path $Stage "runtime\Lib\site-packages") | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $Stage "app") | Out-Null

    # ---- ① 干净安装精简依赖（torch 走 PyPI 即 CPU 版，§4.10 G1）
    $Venv = Join-Path $Root "packaging\pack_venv"
    $Vpy  = Join-Path $Venv "Scripts\python.exe"
    if ($ReuseVenv) {
        if (-not (Test-Path $Vpy)) { throw "-ReuseVenv 需要已存在的 pack_venv，未找到 $Vpy；请去掉该开关全量重建" }
        Step "复用已有构建 venv（-ReuseVenv）"
    } else {
        Step "创建临时构建 venv 并安装精简依赖（首次较慢，torch 约 800 MB）"
        if (Test-Path $Venv) { Remove-Item -Recurse -Force $Venv }
    & $Python -m venv $Venv
        if ($LASTEXITCODE -ne 0) { throw "venv 创建失败，检查 -Python 参数（需 3.12）" }
        $Vpip = Join-Path $Venv "Scripts\pip.exe"
        & $Vpip install --upgrade pip -i $PipIndex --quiet
        & $Vpip install -r (Join-Path $Root "packaging\requirements-pack.txt") `
            --only-binary :all: --no-cache-dir -i $PipIndex
        if ($LASTEXITCODE -ne 0) { throw "依赖安装失败" }
        Smoke $Vpy "venv 原始"
    }

    # ---- ② site-packages 清洗（只删运行时死文件；保留 gradio 前端 .html！）
    Step "清洗 site-packages（tests / doc / 类型桩 / C 源残留）"
    $SP = Join-Path $Venv "Lib\site-packages"
    # 注意：不要删名为 "testing" 的目录——torch\testing、numpy\testing、sympy\testing
    # 都是真正的运行时包（import torch.testing 必经），历史教训见 logs\build_run1.log
    $junkDirNames = @("tests", "test", "docs", "doc", "__pycache__",
                      ".mypy_cache", ".pytest_cache", "benchmarks")
    $junkExts     = @("*.pyi", "*.c", "*.h", "*.hpp", "*.pxd")
    foreach ($d in $junkDirNames) {
        Get-ChildItem $SP -Recurse -Directory -Filter $d -ErrorAction SilentlyContinue |
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    }
    foreach ($e in $junkExts) {
        Get-ChildItem $SP -Recurse -File -Filter $e -ErrorAction SilentlyContinue |
            Remove-Item -Force -ErrorAction SilentlyContinue
    }
    Smoke $Vpy "清洗后"

    # ---- 拷贝 site-packages 到 runtime\（robocopy 多线程，1+ GB 场景远快于 Copy-Item）
    Step "拷贝 site-packages 到 runtime\"
    $Dst = Join-Path $Stage "runtime\Lib\site-packages"
    robocopy $SP $Dst /E /MT:8 /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy 失败（代码 $LASTEXITCODE）" }
    # venv 附带 pip/setuptools 不随包分发（运行期无安装包能力需求）
    foreach ($pat in @("pip*", "setuptools*", "pkg_resources", "wheel*")) {
        Get-ChildItem $Dst -Directory -Filter $pat -ErrorAction SilentlyContinue |
            Remove-Item -Recurse -Force
        Get-ChildItem $Dst -File -Filter $pat -ErrorAction SilentlyContinue |
            Remove-Item -Force
    }
    $spGB = [math]::Round(((Get-ChildItem $Dst -Recurse -File | Measure-Object Length -Sum).Sum / 1GB), 2)
    Info ("精简后 site-packages：{0} GB" -f $spGB)

    # ---- ③ 嵌入式 Python + _pth 修正（§4.4；漏改 _pth = 最常见 ModuleNotFoundError）
    Step "安装嵌入式 Python 到 runtime\"
    $Zip = Join-Path $Logs "embed.zip"
    try {
        Invoke-WebRequest -Uri $EmbedPyUrlCn -OutFile $Zip -UseBasicParsing
    } catch {
        Info "华为云镜像失败，尝试 python.org 官方源"
        Invoke-WebRequest -Uri $EmbedPyUrl -OutFile $Zip -UseBasicParsing
    }
    $pydir = Join-Path $Stage "runtime"
    Expand-Archive -Path $Zip -DestinationPath $pydir -Force
    $pth = Get-ChildItem $pydir -Filter "python*._pth" | Select-Object -First 1
    $content = (Get-Content $pth.FullName) -replace '^\s*#\s*import\s+site\s*$', 'import site'
    # ._pth 中的路径相对 ._pth 所在目录（runtime\）解析，'.' 不是 cwd——
    # 嵌入模式没有 sys.path[0]=当前目录 的行为，必须显式加入 ..\app（build_run3 教训）
    $content += "..\app"
    Set-Content $pth.FullName $content -Encoding Ascii
    Info ("_pth 修正后：" + ($content -join " | "))

    # ---- ④ 汇源码到 app\ + 启动层文件
    Step "复制项目源码到 app\"
    $CodeRoots = @("app.py", "inference.py", "infer_ref_retouch.py", "utils.py",
                   "device_util.py", "llava", "model", "data", "configs", "ui", "data_samples")
    foreach ($c in $CodeRoots) {
        $src = Join-Path $Root $c
        if (Test-Path $src) { Copy-Item $src (Join-Path $Stage "app") -Recurse -Force }
    }
    foreach ($f in @("launcher.py", "download_weights.py", "gpu_setup.py", "VeraRetouch.bat", "开启GPU加速.bat")) {
        Copy-Item (Join-Path $PSScriptRoot $f) $Stage -Force
    }
    Get-ChildItem $Stage -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

    # ---- ⑤ staging 冒烟（不加载权重）
    Step "staging 冒烟：安装目录应可独立 import 全链"
    $Sgpy = Join-Path $Stage "runtime\python.exe"
    Push-Location (Join-Path $Stage "app")
    try {
        & $Sgpy -c "import torch, cv2, transformers, timm, diffusers, gradio; import inference, utils; print('staging import OK')"
        if ($LASTEXITCODE -ne 0) { throw "staging 冒烟失败" }
        & $Sgpy -c "from ui.params_def import SLIDER_KEYS; assert len(SLIDER_KEYS) == 38; print('params def OK')"
        if ($LASTEXITCODE -ne 0) { throw "staging 参数表校验失败" }
    } finally { Pop-Location }
}

# ---------------------------------------------------------------- ⑥ 编译安装器
Step "编译 Inno Setup 安装器"
$Iss = Join-Path $Root "installer\veraretouch.iss"
& $Iscc $Iss
if ($LASTEXITCODE -ne 0) { throw "ISCC 编译失败（若未安装 Inno Setup，请到 innosetup.com 安装后用 -Iscc 指定）" }

# ---------------------------------------------------------------- ⑦ 报告
Step "生成体积与校验报告"
$setup = Get-ChildItem (Join-Path $Root "installer\dist") -Filter "VeraRetouch_Setup*.exe" |
         Sort-Object LastWriteTime -Descending | Select-Object -First 1
$setupMB = [math]::Round($setup.Length / 1MB, 1)
$hash = (Get-FileHash $setup.FullName -Algorithm SHA256).Hash
Info ("安装包：{0}  {1} MB" -f $setup.Name, $setupMB)
Info ("SHA256：{0}" -f $hash)
if ($setupMB -gt 700) { Write-Warning "安装包超过 700 MB 目标，检查精简是否失效" }
Write-Host ""
Write-Host ("完成。产出：" + $setup.FullName) -ForegroundColor Green
} finally { Pop-Location }
