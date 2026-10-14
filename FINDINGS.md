# Findings

Measurements behind the tuner's defaults, and what was tried and dropped. The scripts named below reproduce them.

MiniMax H3 on an RTX 5090. In the ref2va workflow at 768×1344 × 124 frames (dmad full_critic, dpmpp_sde_gpu/beta, 8 steps), dense attention takes 10.4 s per step. One-step errors from verify:

| Allocation | Attention speed-up | Video error | Audio error |
|---|---|---|---|
| fast sinks LoRA (reference) | 4.14× | 14.2% | 14.7% |
| search at the same speed | 4.14× | 13.7% | 14.1% |
| `x3.5_aw1` (balanced) | 3.50× | 10.7% | 11.7% |
| `x3_aw1` (quality) | 3.01× | 8.8% | 10.2% |

Measured end to end with DynamicVRAM, the way a ComfyUI server on NVIDIA loads models: DiT time per clip, averaged over 2 prompts (`tools/compare.py`):

| | dense | fast | x6 |
|---|---|---|---|
| DiT time per clip | 206 s | 85 s | 72 s |
| Speed-up vs dense | 1.00× | 2.41× | 2.84× |

x6 was not singled out in a blind A/B against fast and x8; x8 was, in both clips. The legacy loader runs the next table instead: it partially loads H3 at this size and applies LoRAs to the offloaded INT8 layers as dequantize plus bf16 GEMM, giving 7.1 s per model call against 5.5 s.

With the legacy loader: DiT time per clip, averaged over 2 prompts (`tools/compare.py`):

| | dense | fast | balanced | quality |
|---|---|---|---|---|
| DiT time per clip | 225 s | 113 s | 120 s | 129 s |
| Speed-up vs dense | 1.00× | 2.00× | 1.88× | 1.75× |

Balanced costs about 7 s more per clip than fast, and quality about 16 s more. Going past fast pays off little: the non-attention part of the model dominates by then. In a second run, x6 (6× attention) took 106 s against fast's 113 s, and x8 (7.9×) took 101 s. `dpmpp_sde_gpu` makes about two model calls per step, so per-step estimates from kernel timings understate the difference by about 2×.

Things that didn't beat simply lowering tau on the speed/error frontier at 960×544, dmad 4-step:
- **Kitchen `token_aug`** (64 or 128): about 8% lower error for about 1 ms more per call.
- **3D-tiled video token order**, as in the Veda and Sol papers, with tiles of 4×4×4, 2×4×8 or 1×8×8: about 10% lower error at the same tau, but about 1.8× slower, because more blocks get routed exactly.

Re-tested at the ref2va size (768×1344 × 124 frames) at matched `topk_ratio`, so every order does the same work:
- Tiles cut video error by only 2–4%; for example, 12.3% becomes 11.9% at top 20%.
- The kernel is 7–15% slower even with the reorder cost subtracted, because padding the edge tiles raises the block count from about 550 to 666–705.
- Reordering once per forward pass wouldn't recover that. Untiled order stays.

## Does the tuner measure the right things?

Checked on the SMD Ref2VA 8-step LoRA (euler, its ManualSigmas, 1344×768, two references) with `verify`'s whole-
trajectory pass, 2 prompts:

- **The proxy ranks allocations correctly.** The search's summed per-layer error ranks 8 allocations from 3× to 7×
  the same way as one-step output error (Spearman 0.90) and final-latent error (0.93). On the ref2va dmad stack it
  ranked the five released LoRAs exactly by one-step error (1.00).
- **Speed estimates hold.** Predicted per-step speed-ups were within ~4% of measured DiT time per clip on euler (one
  model call per step), e.g. x6 2.84× predicted, 2.79× measured; dense 102 s, x6 37 s per clip.
- **Final-latent error separates allocations far apart in speed, not ones at a similar speed.** A few-step sampler
  turns any difference into a different, equally valid sample, so even 3× ends 44% from dense. At 960×544 the
  reference LoRA ended closer to dense than ours despite more one-step error. Rank by one-step error and the proxy,
  then check by eye.

## One tune carries across stacks

Per-layer sensitivity on the dmad 8-step stack and on SMD (different LoRA, sampler, schedule, reference count and
orientation) has rank correlation 0.98 for INT8 and 0.93–0.95 for SOL. The x6 tuned on dmad, run on SMD, measured
63.9% / 23.8% final video / audio error against 67.2% / 26.3% for a fresh SMD tune at the same speed: retuning for a
new LoRA on the same base model and size gained nothing. The main per-stack difference is the last layer, which SMD
tolerates about twice as well.

## Released LoRAs, re-measured

Each released file measured in one run per stack (`verify` with allocations read from the files), measured speed-up
per clip and one-step video / audio error:

| Stack | comfyanonymous' test LoRA | ours at the same speed | fastest of ours |
|---|---|---|---|
| ref2va 768×1344, dmad 8-step | 2.19×, 16.4% / 24.6% | balanced 2.20×, 11.1% / 11.3% | x6 2.75×, 22.2% / 19.3% |
| 960×544, dmad 4-step | 1.76×, 15.9% / 12.5% | sinks same-speed 1.75×, 10.0% / 7.4% | sinks fast 1.87×, 14.0% / 6.9% |

At 960×544 × 4 steps the rest of the model is a large share of the step, so every attention LoRA lands at 1.7–1.9×
per clip.
