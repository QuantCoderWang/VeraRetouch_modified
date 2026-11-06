# -*- coding: utf-8 -*-
"""验证 no-cache 补丁的条件请求路径：If-None-Match 匹配 → 304。"""
import urllib.request

URL = "http://127.0.0.1:7899/assets/Index-DRTsT41f.js"
ETAG = '"2dedef8b86d91550cd7a4bac6629b445"'

req = urllib.request.Request(URL, headers={"If-None-Match": ETAG})
try:
    r = urllib.request.urlopen(req)
    print("status:", r.status, "(expected 304 when etag matches)")
except urllib.error.HTTPError as e:
    print("status:", e.code, "(304 = browser revalidates cheaply)")

# ETag 不匹配时必须回 200 新内容
req2 = urllib.request.Request(URL, headers={"If-None-Match": '"stale-etag"'})
r2 = urllib.request.urlopen(req2)
print("mismatch status:", r2.status, "bytes:", len(r2.read()))
