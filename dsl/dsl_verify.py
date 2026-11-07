#!/usr/bin/env python3
"""强验证器: 渲染完成后, 逐场景提取注解完成时刻的帧, OCR复核注解是否真的落在目标文字上。
不达标 = 整体构建失败(拒绝出片)。
用法(系统python3): python3 dsl_verify.py
输入: build/verify_spec.json (渲染器导出: 每条注解的帧文件/期望文字/画的形状屏幕bbox)
"""
import json, os, sys, subprocess

BASE = os.path.dirname(os.path.abspath(__file__))

def main():
    spec_path = os.path.join(BASE, "build", "verify_spec.json")
    if not os.path.exists(spec_path):
        print("[验证跳过] 无注解场景")
        return 0
    spec = json.load(open(spec_path))
    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()
    fails = []
    for entry in spec:
        frame = entry["frame"]  # build/verify_f%d_t%.1f.png
        expect = entry.get("verify_text")
        shape = entry["shape_bbox"]  # 屏幕坐标下我们画的圆/线的包围盒
        if not os.path.exists(frame):
            fails.append((entry, "帧文件缺失")); continue
        if expect:
            res, _ = ocr(frame)
            hit = None
            for r in (res or []):
                if expect in r[1]:
                    box = r[0]
                    xs = [p[0] for p in box]; ys = [p[1] for p in box]
                    hit = (min(xs), min(ys), max(xs), max(ys)); break
            if hit is None:
                fails.append((entry, "帧里OCR找不到目标文字「%s」" % expect)); continue
            # 分类型语义验证
            cx, cy = (hit[0]+hit[2])/2, (hit[1]+hit[3])/2
            etype = entry.get("etype", "circle")
            if etype == "underline":
                # 下划线: x范围与文字对齐, 且紧贴文字下方(0~40px)
                xov = min(hit[2], shape[2]) - max(hit[0], shape[0])
                ok = xov >= 0.8 * (hit[2]-hit[0]) and -8 <= shape[1] - hit[3] <= 45
                why = "FAIL(下划线未对齐文字)" if not ok else ""
            elif etype == "hl":
                # 荧光笔: 覆盖文字, y方向与文字重叠过半
                yov = min(hit[3], shape[3]) - max(hit[1], shape[1])
                ok = cx >= shape[0] and cx <= shape[2] and yov >= 0.5 * (hit[3]-hit[1])
                why = "FAIL(高亮未覆盖文字)" if not ok else ""
            else:
                # 圆圈: 文字中心在圈内(10%边距)
                m = (shape[2]-shape[0])*0.10
                ok = shape[0]-m <= cx <= shape[2]+m and shape[1]-m <= cy <= shape[3]+m
                why = "FAIL(文字中心在形状外)" if not ok else ""
            status = "PASS" if ok else why
            print("[%s] %s | 文字%s 形状%s" % (status, expect, tuple(round(v) for v in hit), tuple(shape)))
            if not ok:
                fails.append((entry, status))
        else:
            # 无文字目标(柱子): 验证帧上形状区域内确实有非背景内容(灰度方差)
            from PIL import Image
            import statistics
            im = Image.open(frame).convert("L").crop([int(v) for v in shape])
            px = list(im.getdata())
            var = statistics.pstdev(px)
            ok = var > 12
            print("[%s] 柱注解 | 区域灰度方差%.1f" % ("PASS" if ok else "FAIL(区域空白)", var))
            if not ok: fails.append((entry, "区域空白"))
    if fails:
        print("\n[验证失败] %d 条不达标, 拒绝出片:" % len(fails))
        for e, why in fails: print("  -", e, why)
        return 1
    print("\n[验证全部通过] 注解均落在目标上 ✓")
    return 0

if __name__ == "__main__":
    sys.exit(main())
