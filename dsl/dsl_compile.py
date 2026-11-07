#!/usr/bin/env python3
"""DSL 编译器: 声明式注解 → 实测坐标。找不到目标 = 硬失败(绝不猜测)。
用法(系统python3, 有rapidocr): python3 dsl_compile.py scene_dsl.json
输入: [{"scene": idx, "image": "xx.png", "crop": null|[..], "effects": [{"type":"circle","text":"-43席","t0":1,"t1":2}, {"type":"hl","chart_bar":1,...}]}]
输出: scenes_resolved.json (供渲染器) —— 每个effect带 bbox
"""
import json, os, sys, subprocess

BASE = os.path.dirname(os.path.abspath(__file__))
MAT = os.path.join(BASE, "materials")
OCR_CACHE = os.path.join(MAT, "ocr_cache.json")

def get_ocr():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()

def ocr_image(ocr, img_name):
    cache = json.load(open(OCR_CACHE)) if os.path.exists(OCR_CACHE) else {}
    if img_name in cache:
        return [tuple(x) for x in cache[img_name]]
    res, _ = ocr(os.path.join(MAT, img_name))
    lines = []
    for r in (res or []):
        if r[2] > 0.5:
            box, text = r[0], r[1]
            xs = [p[0] for p in box]; ys = [p[1] for p in box]
            lines.append((min(xs), min(ys), max(xs), max(ys), text))
    cache[img_name] = lines
    json.dump(cache, open(OCR_CACHE, "w"), ensure_ascii=False)
    return lines

def compile_dsl(dsl_path):
    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()
    cm = json.load(open(os.path.join(MAT, "chart_meta.json"))) if os.path.exists(os.path.join(MAT, "chart_meta.json")) else {}
    scenes = json.load(open(dsl_path))
    out = []
    for sc in scenes:
        img = sc["image"]
        resolved = dict(sc)
        resolved["effects"] = []
        for e in sc.get("effects", []):
            e2 = dict(e)
            if "text" in e2:
                # 意图: 按文字内容定位
                lines = ocr_image(ocr, img)
                cands = [l for l in lines if e2["text"] in l[4]]
                if not cands:
                    # 图表标签也在图里, OCR兜底
                    raise SystemExit("[DSL失败] %s 中找不到文字「%s」——拒绝猜测, 请检查" % (img, e2["text"]))
                x0, y0, x1, y1, _ = cands[0]
                e2["bbox"] = [round(x0), round(y0), round(x1), round(y1)]
                e2["verify_text"] = e2["text"]
            elif "chart_label" in e2:
                key = None
                for k in cm:
                    if k in img: key = k; break
                if not key:
                    raise SystemExit("[DSL失败] %s 无 chart_meta" % img)
                labels = cm[key].get("labels", [])
                hit = [l for l in labels if e2["chart_label"] in str(l)]
                if not hit:
                    raise SystemExit("[DSL失败] %s 的 chart_meta 里找不到标签「%s」" % (img, e2["chart_label"]))
                e2["bbox"] = [round(v) for v in hit[0][:4]] if isinstance(hit[0], (list, tuple)) else hit[0]["bbox"]
                e2["verify_text"] = e2["chart_label"]
            elif "chart_bar" in e2:
                key = None
                for k in cm:
                    if k in img: key = k; break
                bars = cm.get(key, {}).get("bars", [])
                if e2["chart_bar"] >= len(bars):
                    raise SystemExit("[DSL失败] %s bar#%d 不存在" % (img, e2["chart_bar"]))
                e2["bbox"] = [round(v) for v in bars[e2["chart_bar"]]]
                e2.pop("verify_text", None)  # 柱子无文字, 验证走像素
            else:
                raise SystemExit("[DSL失败] effect 必须声明 text/chart_label/chart_bar 之一")
            resolved["effects"].append(e2)
        out.append(resolved)
    json.dump(out, open(os.path.join(BASE, "scenes_resolved.json"), "w"), ensure_ascii=False, indent=1)
    print("[DSL编译通过] %d 场景, %d 个注解全部解析到实测坐标" % (len(out), sum(len(s["effects"]) for s in out)))

if __name__ == "__main__":
    compile_dsl(sys.argv[1])
