# gen.py — 日常声音生成（IndexTTS-2.5 零样本克隆）
# 用法:
#   python gen.py "要念的文案" 输出.wav
#   python gen.py "文案" out.wav --emo happy        # 情绪版
#   python gen.py "文案" out.wav --ref 其他参考.wav  # 换参考音频（切换语气）
#   python gen.py "文案" out.wav --device cpu       # GPU 被占时走 CPU
import argparse
import sys

EMO_ORDER = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", help="要合成的文案")
    ap.add_argument("output", help="输出 wav 路径")
    ap.add_argument("--ref", default="assets/ref_voice.wav"  # 私有资产,不入公开库, help="参考音频(3-10s效果最佳，长音频自动截取前55s亦可)")
    ap.add_argument("--emo", choices=EMO_ORDER, help="情绪控制")
    ap.add_argument("--emo-alpha", type=float, default=0.8, help="情绪强度 0.0-1.0")
    ap.add_argument("--speed", type=float, default=1.0, help="语速因子 0.5-2.0，>1变慢")
    ap.add_argument("--device", default=None, help="cpu / cuda:0，默认自动检测")
    args = ap.parse_args()

    from indextts.infer_v2_5 import IndexTTS2
    import torch

    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    tts = IndexTTS2(
        cfg_path="checkpoints/config.yaml",
        model_dir="checkpoints",
        use_bf16=False,          # V100 不支持 bf16；新卡可改 True 提速
        device=args.device,
    )

    kwargs = dict(
        spk_audio_prompt=args.ref,
        text=args.text,
        lang="ZH",
        output_path=args.output,
        duration_factor=args.speed,
        verbose=True,
    )
    if args.emo:
        v = [0.0] * 8
        v[EMO_ORDER.index(args.emo)] = args.emo_alpha
        kwargs["emo_vector"] = v
        kwargs["use_random"] = False

    tts.infer(**kwargs)
    print(f"[OK] {args.output}")


if __name__ == "__main__":
    sys.exit(main())
