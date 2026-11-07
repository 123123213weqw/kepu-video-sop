#!/usr/bin/env python3
"""从 pass1 的验证帧 OCR 实测文字屏幕坐标 -> calib_screen.json"""
import json, os, sys
from rapidocr_onnxruntime import RapidOCR
BASE = os.path.dirname(os.path.abspath(__file__))
spec = json.load(open(os.path.join(BASE, "build", "verify_spec.json")))
ocr = RapidOCR()
by_text = {}
for e in spec:
    vt = e.get("verify_text")
    if not vt or vt in by_text: continue
    res, _ = ocr(e["frame"])
    for r in (res or []):
        if vt in r[1]:
            box = r[0]
            xs = [p[0] for p in box]; ys = [p[1] for p in box]
            by_text[vt] = [min(xs), min(ys), max(xs), max(ys)]
            break
json.dump(by_text, open(os.path.join(BASE, "calib_screen.json"), "w"), ensure_ascii=False, indent=1)
print("实测屏幕坐标:", json.dumps(by_text, ensure_ascii=False))
for e in spec:
    if e.get("verify_text") and e["verify_text"] not in by_text:
        sys.exit("[抽取失败] " + e["verify_text"])
