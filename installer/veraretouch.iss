; =====================================================================
; VeraRetouch 修图工具 Inno Setup 安装脚本（形态 1：不含模型权重的精简包）
; Per 改造方案 §4.7 / §4.10：
;   - 嵌入式 Python + 精简 site-packages 随包分发（免 Python）
;   - 权重不打包，首次启动由 launcher 经 hf-mirror 下载（形态 2/3 另行安装）
;   - 每用户安装（{localappdata}），无需管理员 / UAC
; 注意：iss 中 "#..." 是 ISPP 预处理指令前缀，注释必须用分号
; =====================================================================

; 形态切换（§4.5）：ISCC 编译时加 /DFULL 即产出含权重的形态 3 全量包
;   iscc installer\veraretouch.iss            → 形态 1：VeraRetouch_Setup.exe（无权重 ~204 MB）
;   iscc /DFULL installer\veraretouch.iss     → 形态 3：VeraRetouch_Full_Setup.exe（含权重，源 staging_full）
#ifdef FULL
#define MyRoot     "..\packaging\staging_full"
#define MyAppName  "VeraRetouch自动修图工具"
#define MyAppExe   "VeraRetouch.bat"
#else
#define MyRoot     "..\packaging\staging"
#define MyAppName  "VeraRetouch自动修图工具"
#define MyAppExe   "VeraRetouch.bat"
#endif

[Setup]
AppId={{A3C7E0F2-9B41-4E8D-9D2A-VERARETOUCH0}}
AppName={#MyAppName}
AppVersion=1.0.0
AppVerName={#MyAppName} 1.0.0
DefaultDirName={localappdata}\VeraRetouch
DefaultGroupName=VeraRetouch
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
Compression=lzma2/max
SolidCompression=no
WizardStyle=modern
OutputDir=.\dist
#ifdef FULL
OutputBaseFilename=VeraRetouch_Full_Setup
#else
OutputBaseFilename=VeraRetouch_Setup
#endif
SetupLogging=yes
UninstallDisplayIcon={app}\{#MyAppExe}

[Languages]
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式(&D)"; \
    GroupDescription: "附加任务："; Flags: checkedonce

[Files]
Source: "{#MyRoot}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExe}"; Tasks: desktopicon
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExe}"

[Run]
Filename: "{app}\{#MyAppExe}"; Description: "立即启动 {#MyAppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; 保留用户产出目录 gradio_outputs 与下载的 checkpoints（卸载时询问由用户手动处理）
Type: filesandordirs; Name: "{app}\app\cache"
