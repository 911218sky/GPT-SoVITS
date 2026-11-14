#!/usr/bin/env python3
"""Run one UVR model on one file, then exit (frees VRAM)."""
import argparse, os, sys, gc
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="weight filename under tools/uvr5/uvr5_weights")
    ap.add_argument("--input", required=True)
    ap.add_argument("--vocal-dir", required=True)
    ap.add_argument("--ins-dir", required=True)
    ap.add_argument("--agg", type=int, default=10)
    ap.add_argument("--format", default="wav")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp16", action="store_true", default=True)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "tools" / "uvr5"))
    os.chdir(root)

    weight = root / "tools" / "uvr5" / "uvr5_weights" / args.model
    if not weight.exists() and not (root / "tools" / "uvr5" / "uvr5_weights" / "onnx_dereverb_By_FoxJoy").exists():
        # allow onnx folder name
        pass
    Path(args.vocal_dir).mkdir(parents=True, exist_ok=True)
    Path(args.ins_dir).mkdir(parents=True, exist_ok=True)

    import torch
    is_half = bool(args.fp16 and args.device.startswith("cuda"))
    model_name = args.model

    if "onnx_dereverb" in model_name:
        from mdxnet import MDXNetDereverb
        pre = MDXNetDereverb(15)
        is_hp3 = False
    elif "roformer" in model_name.lower():
        from bsroformer import Roformer_Loader
        pre = Roformer_Loader(
            model_path=str(weight),
            config_path="",
            device=args.device,
            is_half=is_half,
        )
        # shrink batch for 8GB
        try:
            pre.config["inference"]["batch_size"] = 1
        except Exception:
            pass
        is_hp3 = False
    else:
        from vr import AudioPre, AudioPreDeEcho
        func = AudioPreDeEcho if "DeEcho" in model_name else AudioPre
        pre = func(agg=int(args.agg), model_path=str(weight), device=args.device, is_half=is_half)
        is_hp3 = "HP3" in model_name

    print(f"[uvr_one_stage] model={model_name} in={args.input} half={is_half}", flush=True)
    pre._path_audio_(args.input, args.ins_dir, args.vocal_dir, args.format, is_hp3)
    print("[uvr_one_stage] done", flush=True)

    # explicit free
    del pre
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

if __name__ == "__main__":
    main()
