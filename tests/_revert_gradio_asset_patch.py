# -*- coding: utf-8 -*-
"""一次性回退：把五处 site-packages 的 gradio 前端 accepted_types 恢复为 "image/*"。"""
import re
from pathlib import Path

pat = re.compile(r'accepted_types:"image/\*[^"]*"')
targets = [
    r"D:\VeraRetouch\VeraRetouch-main\vera_venv\Lib\site-packages",
    r"D:\VeraRetouch\VeraRetouch-main\packaging\staging\runtime\Lib\site-packages",
    r"D:\VeraRetouch\VeraRetouch-main\packaging\staging_full\runtime\Lib\site-packages",
    r"D:\veraretouch安装版\VeraRetouch\runtime\Lib\site-packages",
    r"D:\VeraRetouch\VeraRetouch-main\packaging\pack_venv\Lib\site-packages",
]
for t in targets:
    root = Path(t)
    hit = [p for p in root.rglob("*.js") if pat.search(p.read_text(encoding="utf-8", errors="replace"))]
    for p in hit:
        c = p.read_text(encoding="utf-8", errors="replace")
        p.write_text(pat.sub('accepted_types:"image/*"', c), encoding="utf-8")
    print(("reverted " + str(len(hit)) + " file(s) @ ") if hit else "already clean @ ", root.parent.parent.parent)
