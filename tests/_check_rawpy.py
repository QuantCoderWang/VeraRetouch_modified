# -*- coding: utf-8 -*-
"""rawpy 能力体检：版本、体积、API 面。"""
import importlib.metadata as md
import os

import rawpy as rp

d = md.distribution("rawpy")
total = sum(os.path.getsize(str(f)) for f in d.files if os.path.isfile(str(f)))
print("rawpy version:", rp.__version__)
print("installed size MB:", round(total / 1e6, 1))
print("RawPy attrs:", [f for f in ("postprocess", "extract_thumb", "open_file", "sizes")
                       if any(f in dir(x) for x in (rp, rp.RawPy))])
