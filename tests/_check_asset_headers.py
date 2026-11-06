# -*- coding: utf-8 -*-
"""诊断：/assets 响应头（Cache-Control / ETag 是否存在、etag 具体值）。"""
import urllib.request

URL = "http://127.0.0.1:7899/assets/Index-DRTsT41f.js"
r = urllib.request.urlopen(URL)
for k, v in r.headers.items():
    if k.lower() in ("cache-control", "etag", "last-modified", "content-length"):
        print(f"{k}: {v}")
