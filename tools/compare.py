"""Compare attention LoRAs on MiniMax H3: real sampling time for each, and clips for a blind A/B.

For each prompt the conditioning is built once; every config then samples the same seed. DiT time is summed from the
model calls themselves (synchronised, so model loading and VAE work stay out of it). Clips are decoded with video and
audio, saved per config, and copied as A/B/C/... in random order with the key in a separate file.

    <comfy venv>/bin/python tools/compare.py --comfy ~/ComfyUI <stack flags as for tune.py> \\
        --configs dense=,x5=minimax_h3_attn_smd_x5_w1.safetensors,x6=minimax_h3_attn_smd_x6_w1.safetensors --out ab_runs/smd
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import shutil
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attn_tuner.models import ADAPTERS  # noqa: E402
from attn_tuner.runtime import boot_comfy, call, sample  # noqa: E402

AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
# the fp16 video VAE runs out of memory decoding 124 frames at 768x1344 on 32 GB; int8 ConvRot peaks at ~12 GB
DECODE_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"


def main():
    import torch

    adapter = ADAPTERS["minimax_h3"]
    p = argparse.ArgumentParser()
    p.add_argument("--comfy", required=True, type=Path)
    p.add_argument("--configs", required=True, help="name=attention LoRA (empty = no attention LoRA), comma-separated")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--unet")
    p.add_argument("--lora", action="append", default=[])
    p.add_argument("--steps", type=int)
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--sampler")
    p.add_argument("--scheduler")
    p.add_argument("--sigmas", default=None, help="fixed schedule (ManualSigmas), replacing --scheduler / --steps")
    p.add_argument("--prompts", type=int, default=2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--fps", type=float, default=24.0)
    p.add_argument("--decode-only", action="store_true", help="decode the latents a previous run saved")
    p.add_argument("--legacy-vram", action="store_true",
                   help="legacy estimate-based loading instead of DynamicVRAM (a ComfyUI server's default on NVIDIA)")
    adapter.add_args(p)
    args = p.parse_args()
    configs = dict(x.split("=", 1) for x in args.configs.split(","))

    nodes_map = boot_comfy(args.comfy, dynamic_vram=not args.legacy_vram)
    import comfy.model_management

    # decode-only skips the DiT and text encoder, which UNETLoader / CLIPLoader would read into RAM
    base, args.ctx = (None, None) if args.decode_only else adapter.load(nodes_map, args)
    (audio_vae,) = call(nodes_map, "VAELoader", vae_name=AUDIO_VAE)
    (video_vae,) = call(nodes_map, "VAELoader", vae_name=DECODE_VAE)
    models = {}
    for name, lora in ({} if args.decode_only else configs).items():
        m = base
        if lora:
            (m,) = call(nodes_map, "LoraLoaderModelOnly", model=base, lora_name=lora, strength_model=1.0)
        models[name] = m

    calls: list[float] = []

    def wrapper(apply_model, a):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = apply_model(a["input"], a["timestep"], **a["c"])
        torch.cuda.synchronize()
        calls.append(time.perf_counter() - t)
        return out

    args.out.mkdir(parents=True, exist_ok=True)
    timing = {name: [] for name in configs}
    latents = {}  # (prompt, config) -> sampler output; decoded after sampling, with the DiT and text encoder gone
    prompts = list(adapter.eval.items())[: args.prompts]
    saved = args.out.parent / f"{args.out.name}_unblinded" / "latents.pt"  # a failed decode must not cost the sampling again
    if args.decode_only:
        state = torch.load(saved, weights_only=False)
        latents, timing = state["latents"], state["timing"]
        prompts = []
    for pname, prompt in prompts:
        cond, latent = adapter.conditioning(nodes_map, args.ctx, args, prompt)
        # warm-up on the first config: kernel setup and the first model load stay out of the timed runs
        if not latents:
            w = models[next(iter(configs))].clone()
            w.set_model_unet_function_wrapper(wrapper)
            with torch.no_grad():
                sample(nodes_map, w, cond, latent, argparse.Namespace(**{**vars(args), "steps": 2, "sigmas": "1.0, 0.5, 0.0" if args.sigmas else None}), seed=args.seed)
            calls.clear()
        for name, m in models.items():
            m = m.clone()
            m.set_model_unet_function_wrapper(wrapper)
            with torch.no_grad():
                (latents[(pname, name)],) = sample(nodes_map, m, cond, latent, args, seed=args.seed)[:1]
            run = {"prompt": pname, "dit_s": sum(calls), "calls": len(calls), "median_call_s": statistics.median(calls)}
            timing[name].append(run)
            calls.clear()
            print(f"{pname} {name:10s} DiT {run['dit_s']:6.1f} s over {run['calls']} calls "
                  f"({run['median_call_s']:.2f} s/call)", flush=True)
            (args.out.parent / f"{args.out.name}_timing.json").write_text(json.dumps(timing, indent=1))
        del cond, latent
    if not args.decode_only:
        saved.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"latents": latents, "timing": timing}, saved)
        prompts = list(adapter.eval.items())[: args.prompts]
    else:
        prompts = list(dict.fromkeys(p for p, _ in latents))
        prompts = [(p, None) for p in prompts]
        m = w = None

    # unload_all_models alone left ~26 GB allocated during the 124-frame decode; drop every reference instead
    del models, base, m, w
    args.ctx = None
    comfy.model_management.unload_all_models()
    gc.collect()
    comfy.model_management.soft_empty_cache()
    print(f"before decode: {torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated", flush=True)

    key, rng = {}, random.Random()
    for pname, _ in prompts:
        files = {}
        for name in configs:
            out = latents.pop((pname, name))
            # ComfyUI's executor runs nodes under inference_mode; without it the decoded audio carries grad
            with torch.inference_mode():
                (images,) = call(nodes_map, "VAEDecode", samples=out, vae=video_vae)
                (audio,) = call(nodes_map, "VAEDecodeAudio", samples=out, vae=audio_vae)
                (video,) = call(nodes_map, "CreateVideo", images=images, fps=args.fps, audio=audio)
                files[name] = saved.parent / f"{pname}_{name}.mp4"  # named copies stay out of the blind folder
                files[name].parent.mkdir(exist_ok=True)
                video.save_to(str(files[name]))
            del out, images, audio, video
        order = list(configs)
        rng.shuffle(order)
        for letter, name in zip("ABCDEFGH", order):
            shutil.copy(files[name], args.out / f"{pname}_{letter}.mp4")
        key[pname] = dict(zip("ABCDEFGH", order))
        (args.out.parent / f"{args.out.name}_KEY.json").write_text(json.dumps({"key": key, "timing": timing}, indent=1))

    print("DiT seconds per clip (median over prompts):")
    ref = statistics.median(r["dit_s"] for r in next(iter(timing.values())))
    for name, runs in timing.items():
        t = statistics.median(r["dit_s"] for r in runs)
        print(f"  {name:10s} {t:6.1f} s  ({ref / t:.2f}x vs {next(iter(configs))})")
    print(f"blind clips in {args.out}; key in {args.out.name}_KEY.json (don't open it until you've picked)")


if __name__ == "__main__":
    main()
