# comfy-attn-tuner

Makes MiniMax H3 sample faster in ComfyUI, with no retraining. It measures every attention layer of your exact workflow
(model, LoRAs, sampler, resolution), gives each layer the fastest attention kernel it tolerates, and saves the result
as a LoRA you load like any other.

On an RTX 5090, H3 ref2va at 768×1344, 124 frames: **198 s → 72 s** of sampling per clip (2.75×). In blind A/B
tests it wasn't picked out from slower attention LoRAs.

## Ready-made LoRAs

LoRAs already tuned for common H3 stacks are on Hugging Face:
[nynxz/MiniMax-H3-Experiments](https://huggingface.co/nynxz/MiniMax-H3-Experiments) (`loras/`, with the measurements in
[docs/attention.md](https://huggingface.co/nynxz/MiniMax-H3-Experiments/blob/main/docs/attention.md)). Run the tuner
when your stack differs.

## Requirements

- An NVIDIA Blackwell GPU (RTX 50-series or B-series). The comfy-kitchen kernels need one.
- ComfyUI with attention configs in LoRAs ([Comfy-Org/ComfyUI#16994](https://github.com/Comfy-Org/ComfyUI/pull/16994)).
- ComfyUI's own Python. Stop the ComfyUI server first: the tuner needs the whole GPU.

## Quick start

Tune for your stack in four steps. `profile` and `verify` take the same stack flags. This example is the
[SMD Ref2VA 8-step](https://huggingface.co/zjwfufu/SMD-MiniMax-H3-Ref2VA-8step-ComfyUI) workflow:

```bash
PY=~/ComfyUI/.venv/bin/python
STACK=(--model minimax_h3 --comfy ~/ComfyUI --tag smd
       --unet minimax_h3_ref2va_pruned_int8_convrot.safetensors --lora SMD-MiniMax-H3-Ref2VA-8step.safetensors
       --sampler euler --steps 8 --shift 12,3
       --sigmas "1.0, 0.9873417722, 0.9705882353, 0.9473684211, 0.9130434783, 0.8571428571, 0.75, 0.4615384615, 0.0"
       --width 1344 --height 768 --length 124
       --ref-image person.png --ref-image object.png --ref-short-edge 512 --ref2va-prompts)

$PY tune.py profile "${STACK[@]}" --sinks                    # measures every layer x kernel (the long step)
$PY tune.py search --model minimax_h3 --tag smd --targets 3 4 5 6
$PY tune.py verify "${STACK[@]}" --only x4_w1,x5_w1,x6_w1    # real error and real time for each pick
$PY tune.py write --model minimax_h3 --tag smd --pick x5_w1  # saves the LoRA
```

## Using the LoRA

`write` saves `results/minimax_h3/minimax_h3_attn_<tag>_<pick>.safetensors`. Copy it into `ComfyUI/models/loras` and
add a **LoraLoaderModelOnly** at strength 1.0, last in the model chain. Don't combine it with attention-override nodes.

A LoRA is tuned for one stack. It still works if you change the resolution, length, steps or LoRAs, but may no longer be
the best trade-off; re-run the tuner for a stack you use a lot.

## Choosing a pick

`search` prints one line per pick:

| Name | Meaning |
|---|---|
| `x4` | 4× faster attention, lowest overall error |
| `x4_w1`, `x4_w3`, `x4_w10` | the same speed, with audio error weighted 1, 3 or 10× against video |
| `ref_speed_*` | the speed of `--reference-lora`, to compare against a LoRA you already use |

`verify` then runs each pick for real and prints:

- **one-step error**: the output error of a single model call;
- **final error**: how far the finished latent ends up from the dense one on the same seed. Small changes early on
  send a few-step sampler to a different but equally valid result, so even near-lossless picks show large numbers
  here; compare picks against each other, not against zero;
- **DiT time** per clip, and the speed-up over dense.

Take the fastest pick whose final error is close to that of a LoRA you're already happy with, then check it by eye.
For H3, attention beyond about 6× stops paying: the rest of the model dominates, and 8× was visibly different in a
blind test.

## Options

| Flag | |
|---|---|
| `--tag` | names this stack's result files |
| `--unet`, `--lora` (repeatable), `--sampler`, `--scheduler`, `--steps`, `--sigmas`, `--shift` | the sampling stack |
| `--width`, `--height`, `--length` | output size |
| `--ref-image` (repeatable), `--ref-short-edge` | ref2va references, in `<Picture N>` order |
| `--ref2va-prompts` | tune on structured two-reference prompts (a person and an object) instead of plain text-to-video prompts |
| `--sinks` | also try SOL with the prefix keys and audio queries kept exact (recommended) |
| `--prompts` | prompts per phase (default 2) |
| `--targets` | attention speed-ups for `search` to pick at (default 3 3.5 4) |
| `--reference-lora` | an existing attention LoRA for `search` and `verify` to compare against |
| `--per-step` | allow a different kernel per denoising step |
| `--no-trajectory` | `verify` only measures one-step error (faster) |

## Comparing LoRAs

`tools/compare.py` samples a few attention LoRAs (and dense) with your stack, reports the real sampling time of
each, and saves the clips as a blind A/B/C set with the key in a separate file. Use it to check a pick by eye.

## How it works

Each layer can run dense attention, comfy-kitchen INT8 attention, or comfy-kitchen SOL sparse attention
([Sol-Attn](https://arxiv.org/abs/2607.24027)) at several sparsity levels. `profile` samples your stack and, at every
attention call, runs every candidate on the same inputs, recording its error against dense and its time. `search`
solves a knapsack over layers for the lowest total error at each speed. `verify` confirms the picks by sampling with
them. Measurements and the ideas that didn't pan out are in [FINDINGS.md](FINDINGS.md).

CPU tests for the search and the LoRA format: `<ComfyUI venv>/bin/python -m pytest tests`.

To add a model, subclass `attn_tuner/models/base.py:Adapter` and register it in `attn_tuner/models/__init__.py`.

## Credits

- The INT8 and SOL kernels are from comfy-kitchen; SOL follows Sol-Attn (arXiv 2607.24027).
- The config-LoRA format follows comfyanonymous' `fast_sol_kitchen_attention` test LoRA.
