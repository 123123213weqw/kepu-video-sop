#!/usr/bin/env python3
"""字幕引擎 v2: faster-whisper 强制对齐 -> 句级 .ass 字幕
用法: python sub_engine.py <旁白完整wav> <chunks.json> <输出.ass>
依赖: GPT-SoVITS/.venv 的 faster-whisper + 本地模型目录 fwhisper/
"""
import sys, json, re, subprocess

MODEL_DIR = "./fwhisper"  # faster-whisper small 本地模型目录
PY = "python"

def chars(s):
    return re.sub(r"[^\u4e00-\u9fff0-9a-zA-Z]", "", s)

def sec2ts(t):
    h = int(t // 3600); m = int(t % 3600 // 60); s = t % 60
    return "%d:%02d:%02d.%02d" % (h, m, int(s), int((s % 1) * 100))

def main(wav, chunks_json, out_ass, offset=0.0):
    from faster_whisper import WhisperModel
    model = WhisperModel(MODEL_DIR, device="cpu", compute_type="int8")
    chunks = json.load(open(chunks_json))
    texts = [s.strip() for c in chunks for s in re.split(r"(?<=[。？！])", c) if s.strip()]

    # 逐段对齐(段内时间从0起, 加段偏移)
    events = []
    # 先测各段时长拼完整时间轴 -> 这里简化: 对完整拼接wav一次对齐, 按全稿句子贪心匹配
    segments, _ = model.transcribe(wav, language="zh", word_timestamps=True, vad_filter=False)
    words = []
    for seg in segments:
        for w in (seg.words or []):
            words.append((w.start, w.end, w.word.strip()))
    idx = 0
    for s in texts:
        need = len(chars(s))
        cnt, st, en = 0, None, None
        while idx < len(words) and cnt < need:
            c = len(chars(words[idx][2]))
            if c == 0:
                idx += 1; continue
            if st is None: st = words[idx][0]
            en = words[idx][1]
            cnt += c
            idx += 1
        if st is None: continue
        # 长句按逗号拆两行(时间按字数比例)
        lines = [s]
        if len(s) > 16 and "，" in s:
            mid = s.rfind("，", 0, int(len(s) * 0.65))
            if mid > 4:
                a, b = s[:mid+1], s[mid+1:]
                ra = len(chars(a)) / max(1, len(chars(s)))
                mid_t = st + (en - st) * ra
                lines = [(a, st, mid_t), (b, mid_t, en)]
        if len(lines) == 1: lines = [(s, st, en)]
        for txt, t0, t1 in lines:
            events.append((t0 + offset, t1 + offset, txt))

    ass = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sub,Noto Sans SC,52,&H00FFFFFF,&H00000000,&H80000000,0,3,0,2,60,60,40,1

[Events]
Format: Layer, Start, End, Style, Text
"""
    for st, en, txt in events:
        ass += "Dialogue: 0,%s,%s,Sub,%s\n" % (sec2ts(st), sec2ts(en), txt.replace("\n", " "))
    open(out_ass, "w", encoding="utf8").write(ass)
    print("字幕 %d 条 -> %s" % (len(events), out_ass))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 0.0)
