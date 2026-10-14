"""Search per-layer attention kernels (dense / kitchen INT8 / kitchen SOL) for a ComfyUI model stack, and write the
winners as attention-config LoRAs.

Run with ComfyUI's Python on a Blackwell GPU (the kitchen kernels need it), with no ComfyUI server sharing the GPU.

  profile  Generate with the exact stack. At every DiT self-attention call the dense result is kept, and each candidate
           also runs on the same q/k/v: error (whole output and per row group, e.g. video / audio) and kernel time per
           (layer, step, candidate). One extra plain generation times a dense step, for end-to-end estimates.
  search   Exact knapsack over layers (or, with --per-step, over (layer, step) pairs): least error for each
           attention-time budget, optionally weighting the secondary row groups; named picks at target speed-ups and
           at the speed of a reference config LoRA.
  verify   Generate again; at every step the model also runs under each allocation on identical inputs, giving the
           one-step output error of each allocation per output group. Then each allocation samples the whole
           trajectory on the same seed: final-latent error against dense, which is what reaches the decoded output,
           and measured DiT time.
  write    Save chosen allocations as config LoRAs (diffusion_model.blocks.N.attn.comfy_attention.config).

--sinks adds SOL candidates that keep the model's sink rows exact, and --per-step writes per-step configs
(sigma_range lists); both need ComfyUI's attention-config extension (Comfy-Org/ComfyUI#16994).

    python tune.py profile --model minimax_h3 --comfy ~/ComfyUI --tag mystack <stack flags> --sinks
    python tune.py search  --model minimax_h3 --tag mystack --targets 4 5 6
    python tune.py verify  --model minimax_h3 --comfy ~/ComfyUI --tag mystack <stack flags> --only x4_w1,x6_w1
    python tune.py write   --model minimax_h3 --tag mystack --pick x6_w1

See README.md for a complete stack.
"""

from __future__ import annotations

import argparse
import functools
import json
import statistics
import time
from pathlib import Path

from .models import ADAPTERS
from .runtime import StepOf, boot_comfy, rel, sample

CANDIDATES = {  # name -> attention config (None = dense, ComfyUI's default path)
    "dense": None,
    "int8": {"attention": "comfy_kitchen_int8"},
    # tau 1.3 is the kernel default; 1.3 and 2.0 also appear in comfyanonymous' reference config LoRA.
    **{f"sol{t:g}": {"attention": "comfy_kitchen_sol", "tau": t} for t in (0.0, 0.5, 1.0, 1.3, 1.5, 2.0)},
}
SINK_TAUS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)


def candidates(adapter, sinks: bool) -> dict:
    extra = {f"sol{t:g}s": {"attention": "comfy_kitchen_sol", "tau": t, **adapter.sinks} for t in SINK_TAUS}
    return {**CANDIDATES, **(extra if sinks else {})}


def attention_fns(cands: dict) -> dict:
    from comfy.ldm.modules import attention as A

    fns = {}
    for name, cfg in cands.items():
        if cfg is None:
            fns[name] = A.attention_pytorch
        elif cfg["attention"] == "comfy_kitchen_int8":
            fns[name] = A.attention_comfy_kitchen_int8
        else:
            fns[name] = functools.partial(A.attention_comfy_kitchen_sol, **{k: v for k, v in cfg.items() if k != "attention"})
    return fns


def dit_call(adapter, q, kw):
    """(row spans per group, kwargs for the candidates) for a DiT self-attention call, else None."""
    to = kw.get("transformer_options", {})
    if q.shape[-2] < adapter.min_tokens:
        return None
    rows = adapter.rows(to)
    if rows is None:
        return None
    return rows, {**kw, "transformer_options": adapter.spans(to)}


def time_dense_step(nodes_map, adapter, model, args) -> float:
    """Median seconds per dense DiT step (model call only; the first step, which loads the model, is dropped)."""
    import torch

    times = []

    def wrapper(apply_model, a):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = apply_model(a["input"], a["timestep"], **a["c"])
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
        return out

    m = model.clone()
    m.set_model_unet_function_wrapper(wrapper)
    cond, latent = adapter.conditioning(nodes_map, args.ctx, args, adapter.calib[0])
    with torch.no_grad():
        sample(nodes_map, m, cond, latent, args, seed=1000)
    return statistics.median(times[1:] or times)


def profile(adapter, args):
    import torch

    nodes_map = boot_comfy(args.comfy, dynamic_vram=not args.legacy_vram)
    cands = candidates(adapter, args.sinks)
    fns = attention_fns(cands)
    step_of = StepOf()
    stats: dict = {}  # layer -> step -> candidate -> lists
    keys = ["err"] + [f"err_{g}" for g in adapter.groups]

    def rows_err(out, ref, spans):
        if len(spans) == 1:
            (a, b), = spans
            return rel(out[:, a:b], ref[:, a:b])
        return rel(torch.cat([out[:, a:b] for a, b in spans], dim=1), torch.cat([ref[:, a:b] for a, b in spans], dim=1))

    def override(func, q, k, v, heads, **kw):
        ref = fns["dense"](q, k, v, heads, **kw)
        call = dit_call(adapter, q, kw)
        if call is None:
            return ref
        rows, ckw = call
        row = stats.setdefault(str(kw["transformer_options"]["block_index"]), {}).setdefault(str(step_of(kw)), {})
        for name, fn in fns.items():
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn(q, k, v, heads, **ckw)
            end.record()
            torch.cuda.synchronize()
            cell = row.setdefault(name, {**{k_: [] for k_ in keys}, "ms": []})
            dense = name == "dense"
            cell["err"].append(0.0 if dense else rel(out, ref))
            for g, spans in rows.items():
                cell[f"err_{g}"].append(0.0 if dense else rows_err(out, ref, spans))
            cell["ms"].append(start.elapsed_time(end))
            del out
        return ref

    base, args.ctx = adapter.load(nodes_map, args)
    step_s = time_dense_step(nodes_map, adapter, base, args) if args.time_steps else None
    if step_s:
        print(f"dense step: {step_s:.2f} s")
    prompts = adapter.calib[: args.prompts]
    model = base.clone()
    model.model_options["transformer_options"]["optimized_attention_override"] = override
    for p_i, prompt in enumerate(prompts):
        cond, latent = adapter.conditioning(nodes_map, args.ctx, args, prompt)
        with torch.no_grad():
            sample(nodes_map, model, cond, latent, args, seed=1000 + p_i)
        print(f"profiled prompt {p_i + 1}/{len(prompts)}: {len(stats)} layers", flush=True)

    def summarize(lists_by_cand):
        return {c: {**{k_: statistics.mean(v[k_]) for k_ in keys}, "ms": statistics.median(v["ms"]), "n": len(v["err"])}
                for c, v in lists_by_cand.items()}

    def merge(step_rows):
        merged: dict = {}
        for row in step_rows.values():
            for c, v in row.items():
                m = merged.setdefault(c, {k_: [] for k_ in v})
                for k_ in v:
                    m[k_] += v[k_]
        return merged

    layers = {l: summarize(merge(steps)) for l, steps in sorted(stats.items(), key=lambda kv: int(kv[0]))}
    per_step = {f"{l}:{s}": summarize(row) for l, steps in stats.items() for s, row in steps.items()}
    meta = {"model": adapter.name,
            **{k_: v for k_, v in vars(args).items() if k_ not in ("comfy", "ctx", "phase", "results")},
            "prompts": len(prompts), "candidates": cands, "sample_sigmas": step_of.schedule, "dense_step_s": step_s}
    out_dir(args).mkdir(parents=True, exist_ok=True)
    (out_dir(args) / f"{args.tag}_profile.json").write_text(
        json.dumps({"meta": meta, "layers": layers, "per_step": per_step}, indent=1))
    dense = sum(r["dense"]["ms"] for r in layers.values())
    print(f"dense attention per step: {dense / 1e3:.2f} s over {len(layers)} layers"
          + (f" ({dense / 1e3 / step_s * 100:.0f}% of the step)" if step_s else ""))
    for c in cands:
        t = sum(r[c]["ms"] for r in layers.values())
        errs = " ".join(f"{g} {statistics.mean(r[c][f'err_{g}'] for r in layers.values()):.4f}" for g in adapter.groups)
        e2e = f", step {step_s / (step_s - (dense - t) / 1e3):.2f}x" if step_s else ""
        print(f"  all-{c:7s}: {t / 1e3:.2f} s/step attention ({dense / t:.2f}x{e2e}), mean layer rel err {errs}")


def out_dir(args) -> Path:
    return Path(args.results) / args.model


def load_reference(adapter, path: str | None) -> dict | None:
    if not path:
        return None
    from safetensors.torch import load_file

    prefix, suffix = adapter.config_key("{}").split("{}")
    alloc = {}
    for key, t in load_file(path).items():
        # DiT blocks only: e.g. H3's token_refiner.blocks.N would otherwise overwrite DiT block N.
        if key.startswith(prefix) and key.endswith(suffix):
            layer = key[len(prefix):-len(suffix)]
            cfg = json.loads(bytes(t.tolist()))
            name = "int8" if cfg["attention"] == "comfy_kitchen_int8" else f"sol{cfg.get('tau', 1.0):g}"
            if "sink_keys" in cfg:
                name += "s"
            alloc[layer] = name
    return alloc


def cell_err(cell: dict, groups: tuple, weight: float | None) -> float:
    """Per-item proxy: whole-output error, or primary-group error + weight * the other groups' error."""
    if weight is None:
        return cell["err"]
    return cell[f"err_{groups[0]}"] + weight * sum(cell[f"err_{g}"] for g in groups[1:])


def cost(items: dict, alloc: dict, groups: tuple, weight: float | None = None) -> tuple[float, float]:
    t = sum(items[k][alloc.get(k, "dense")]["ms"] for k in items)
    e = sum(cell_err(items[k][alloc.get(k, "dense")], groups, weight) for k in items)
    return t, e


def frontier(items: dict, groups: tuple, weight: float | None, unit_ms: float = 2.0) -> list[tuple[float, float, dict]]:
    """Exact min-error allocation for every attention-time budget (knapsack over items, time in ``unit_ms``)."""
    best = {0: (0.0, {})}  # time units -> (error, allocation)
    for key in sorted(items):
        nxt = {}
        for units, (err, alloc) in best.items():
            for c, cell in items[key].items():
                u = units + round(cell["ms"] / unit_ms)
                e = err + cell_err(cell, groups, weight)
                if u not in nxt or e < nxt[u][0]:
                    nxt[u] = (e, {**alloc, key: c})
        best, low = {}, float("inf")
        for u in sorted(nxt):  # keep Pareto-optimal states only
            if nxt[u][0] < low:
                best[u], low = nxt[u], nxt[u][0]
    return [(u * unit_ms, e, {k: c for k, c in a.items() if c != "dense"}) for u, (e, a) in sorted(best.items())]


def search(adapter, args):
    prof = json.loads((out_dir(args) / f"{args.tag}_profile.json").read_text())
    items = prof["per_step"] if args.per_step else prof["layers"]
    groups = adapter.groups
    step_s = prof["meta"].get("dense_step_s")
    dense_t, _ = cost(items, {}, groups)
    out = {"dense_ms": dense_t, "dense_step_s": step_s, "per_step": args.per_step, "allocations": {}}

    def add(name, alloc):
        t, e = cost(items, alloc, groups)
        proxy = {g: sum(items[k][alloc.get(k, "dense")][f"err_{g}"] for k in items) for g in groups}
        # attention times are per step when searching per layer, and summed over steps when searching per step
        steps = len({k.split(":")[1] for k in items}) if args.per_step else 1
        e2e = step_s / (step_s - (dense_t - t) / 1e3 / steps) if step_s else None
        out["allocations"][name] = {"alloc": alloc, "attn_ms": t, "proxy_err": e,
                                    **{f"proxy_{g}": v for g, v in proxy.items()}, "speedup": dense_t / t,
                                    "step_speedup": e2e}
        counts = {c: list(alloc.values()).count(c) for c in sorted(set(alloc.values()))}
        errs = " ".join(f"{g} {v:.3f}" for g, v in proxy.items())
        print(f"{name:22s} attention {dense_t / t:.2f}x" + (f", step {e2e:.2f}x" if e2e else "")
              + f", proxy {errs}, {counts}")

    # secondary-group weights mean nothing for a single-group model
    weights = [None] + (list(args.weights) if args.weights is not None else [1.0, 3.0, 10.0] if len(groups) > 1 else [])
    # time unit: 2 ms suits H3's ~200 ms calls; models with short calls need finer steps to hit a target exactly
    unit = min(2.0, dense_t / 4000)
    fronts = {w: frontier(items, groups, w, unit) for w in weights}

    def best_under(w, t_max):
        ok = [p for p in fronts[w] if p[0] <= t_max]
        return min(ok, key=lambda p: p[1])[2] if ok else None

    ref = load_reference(adapter, args.reference_lora)
    targets = [(f"x{x:g}", dense_t / x) for x in args.targets]
    if ref is not None:
        if args.per_step:  # the reference is per layer: the same kernel at every step
            steps = sorted({k.split(":")[1] for k in items})
            ref = {f"{l}:{s}": c for l, c in ref.items() for s in steps}
        add("reference", ref)
        targets = [("ref_speed", cost(items, ref, groups)[0])] + targets
    for label, t_max in targets:
        for w in weights:
            alloc = best_under(w, t_max)
            if alloc is not None:
                add(label + ("" if w is None else f"_w{w:g}"), alloc)
    (out_dir(args) / f"{args.tag}_search.json").write_text(json.dumps(out, indent=1))


def verify(adapter, args):
    import torch

    prof = json.loads((out_dir(args) / f"{args.tag}_profile.json").read_text())
    allocs = json.loads((out_dir(args) / f"{args.tag}_search.json").read_text())["allocations"]
    if args.only:  # checked before ComfyUI loads anything
        unknown = set(args.only.split(",")) - set(allocs)
        if unknown:
            raise SystemExit(f"--only: no allocation named {sorted(unknown)}; search wrote {sorted(allocs)}")
        allocs = {k: v for k, v in allocs.items() if k in args.only.split(",")}
    nodes_map = boot_comfy(args.comfy, dynamic_vram=not args.legacy_vram)
    fns = attention_fns(prof["meta"]["candidates"])
    step_of = StepOf()
    state = {"alloc": {}}
    errs = {name: {g: [] for g in adapter.groups} for name in allocs}
    final = {name: {g: [] for g in adapter.groups} for name in allocs}  # final-latent error after the whole trajectory
    dit_s = {name: [] for name in ["dense", *allocs]}

    def override(func, q, k, v, heads, **kw):
        call = dit_call(adapter, q, kw)
        if call is None:
            return fns["dense"](q, k, v, heads, **kw)
        layer = kw["transformer_options"]["block_index"]
        alloc = state["alloc"]
        choice = alloc.get(f"{layer}:{step_of(kw)}") or alloc.get(str(layer)) or "dense"
        return fns[choice](q, k, v, heads, **call[1])

    def wrapper(apply_model, a):
        x, t, c = a["input"], a["timestep"], a["c"]
        state["alloc"] = {}
        ref = apply_model(x, t, **c)
        r = adapter.split_output(ref, c)
        for name, spec in allocs.items():
            state["alloc"] = spec["alloc"]
            o = adapter.split_output(apply_model(x, t, **c), c)
            for g in adapter.groups:
                errs[name][g].append(rel(o[g], r[g]))
        state["alloc"] = {}
        return ref

    base, args.ctx = adapter.load(nodes_map, args)
    model = base.clone()
    model.model_options["transformer_options"]["optimized_attention_override"] = override
    model.set_model_unet_function_wrapper(wrapper)
    calls: list[float] = []

    def timed(apply_model, a):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = apply_model(a["input"], a["timestep"], **a["c"])
        torch.cuda.synchronize()
        calls.append(time.perf_counter() - t)
        return out

    traj = base.clone()
    traj.model_options["transformer_options"]["optimized_attention_override"] = override
    traj.set_model_unet_function_wrapper(timed)
    for pname, prompt in list(adapter.eval.items())[: args.prompts]:
        cond, latent = adapter.conditioning(nodes_map, args.ctx, args, prompt)
        with torch.no_grad():
            sample(nodes_map, model, cond, latent, args, seed=args.seed)
            if args.trajectory:
                outs = {}
                for name in dit_s:
                    state["alloc"] = allocs[name]["alloc"] if name != "dense" else {}
                    calls.clear()
                    outs[name] = sample(nodes_map, traj, cond, latent, args, seed=args.seed)[0]["samples"]
                    dit_s[name].append(sum(calls))
                state["alloc"] = {}
                ref = adapter.split_latent(outs.pop("dense"))
                for name, x in outs.items():
                    o = adapter.split_latent(x)
                    for g in adapter.groups:
                        final[name][g].append(rel(o[g], ref[g]))
                del outs, ref
        print(f"verified {pname}", flush=True)
    result = {}
    for name, e in errs.items():
        r = {**{g: statistics.mean(v) for g, v in e.items()}, "speedup": allocs[name]["speedup"],
             "step_speedup": allocs[name].get("step_speedup"), "proxy_err": allocs[name]["proxy_err"]}
        if args.trajectory:
            r |= {f"final_{g}": statistics.mean(v) for g, v in final[name].items()}
            r |= {"dit_s": statistics.mean(dit_s[name]), "measured_speedup": statistics.mean(dit_s["dense"])
                  / statistics.mean(dit_s[name])}
        result[name] = r
    (out_dir(args) / f"{args.tag}_verify.json").write_text(json.dumps(result, indent=1))
    if args.trajectory:
        print(f"dense: DiT {statistics.mean(dit_s['dense']):.1f} s per clip")
    for name, r in result.items():
        e2e = f", step {r['step_speedup']:.2f}x" if r["step_speedup"] else ""
        errs_s = " ".join(f"{g} {r[g] * 100:.1f}%" for g in adapter.groups)
        line = f"{name:22s} attention {r['speedup']:.2f}x{e2e} | one-step error {errs_s}"
        if args.trajectory:
            line += (" | final " + " ".join(f"{g} {r[f'final_{g}'] * 100:.1f}%" for g in adapter.groups)
                     + f" | DiT {r['dit_s']:.1f} s ({r['measured_speedup']:.2f}x)")
        print(line)
    # does the search's summed per-layer proxy rank allocations the way the outputs do? (allocations built from
    # LoRAs that use kernels the profile did not measure have no proxy)
    if len(result) >= 3 and all(r["proxy_err"] is not None for r in result.values()):
        names = list(result)
        for key in ["one-step", *(["final"] if args.trajectory else [])]:
            g = adapter.groups[0]
            col = [result[n][g if key == "one-step" else f"final_{g}"] for n in names]
            print(f"proxy vs {key} {g} error, Spearman rank correlation over {len(names)} allocations: "
                  f"{spearman([result[n]['proxy_err'] for n in names], col):.2f}")


def spearman(a: list[float], b: list[float]) -> float:
    def ranks(x):
        order = sorted(range(len(x)), key=x.__getitem__)
        r = [0.0] * len(x)
        for i, j in enumerate(order):
            r[j] = float(i)
        return r

    ra, rb = ranks(a), ranks(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    return cov / (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5


def layer_configs(alloc: dict, cands: dict, schedule: list[float]) -> dict:
    """Per-layer config(s); per-step allocations become sigma_range lists split halfway between schedule sigmas."""
    if not any(":" in k for k in alloc):
        return {l: cands[c] for l, c in alloc.items()}
    steps = len(schedule) - 1
    bounds = [(0.5 * (schedule[i] + schedule[i + 1]) if i + 1 < steps else 0.0,
               0.5 * (schedule[i - 1] + schedule[i]) if i > 0 else 1e9) for i in range(steps)]
    layers = sorted({k.split(":")[0] for k in alloc}, key=int)
    out = {}
    for l in layers:
        picks = [alloc.get(f"{l}:{s}", "dense") for s in range(steps)]
        if len(set(picks)) == 1:
            if picks[0] != "dense":
                out[l] = cands[picks[0]]
            continue
        out[l] = [{**(cands[c] or {"attention": "dense"}), "sigma_range": [lo, hi]}
                  for c, (lo, hi) in zip(picks, bounds)]
    return out


def public_stack(meta: dict) -> dict:
    """The profile's stack settings for LoRA metadata: the number of reference images, not their local paths."""
    stack = {k: v for k, v in meta.items() if k not in ("candidates", "ref_image")}
    refs = meta.get("ref_image") or []
    if refs:
        stack["ref_images"] = 1 if isinstance(refs, str) else len(refs)  # older profiles: one path as a str
    return stack


def write(adapter, args):
    import torch
    from safetensors.torch import save_file

    d = out_dir(args)
    allocs = json.loads((d / f"{args.tag}_search.json").read_text())["allocations"]
    verified = json.loads((d / f"{args.tag}_verify.json").read_text()) if (d / f"{args.tag}_verify.json").exists() else {}
    prof_meta = json.loads((d / f"{args.tag}_profile.json").read_text())["meta"]
    for name in args.pick.split(","):
        alloc = allocs[name]["alloc"]
        configs = layer_configs(alloc, prof_meta["candidates"], prof_meta["sample_sigmas"])
        tensors = {adapter.config_key(l): torch.tensor(list(json.dumps(cfg).encode()), dtype=torch.uint8)
                   for l, cfg in sorted(configs.items(), key=lambda kv: int(kv[0]))}
        counts = {c: list(alloc.values()).count(c) for c in set(alloc.values())}
        lora = prof_meta.get("lora") or []
        loras = " + ".join([lora] if isinstance(lora, str) else lora) or "no LoRA"  # older profiles: one str
        meta = {"description": f"{adapter.title} attention allocation '{name}' tuned for {prof_meta['unet']} + {loras}, "
                               f"{prof_meta['steps']} steps at {prof_meta['width']}x{prof_meta['height']}",
                "usage": "Load LoRA (Model Only), strength_model=1.0, last in the stack; disable attention override nodes",
                "requires": "ComfyUI attention-config extension (sinks / per-step)"
                            if any(isinstance(c, list) or "sink_keys" in (c or {}) for c in configs.values()) else "",
                "attention_counts": json.dumps(counts),
                "measured": json.dumps(verified.get(name, {})),
                "stack": json.dumps(public_stack(prof_meta))}
        path = d / f"{adapter.name}_attn_{args.tag}_{name}.safetensors"
        save_file(tensors, path, metadata=meta)
        print(path, counts)


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--model", choices=sorted(ADAPTERS), required=True)
    adapter = ADAPTERS[pre.parse_known_args()[0].model]

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                parents=[pre])
    p.add_argument("phase", choices=["profile", "search", "verify", "write"])
    p.add_argument("--comfy", type=Path, default=Path.home() / "ComfyUI", help="ComfyUI checkout (profile, verify)")
    p.add_argument("--tag", required=True)
    p.add_argument("--results", default=Path(__file__).resolve().parents[1] / "results")
    p.add_argument("--unet")
    p.add_argument("--lora", action="append", default=[], help="model-only LoRA at strength 1; repeatable")
    p.add_argument("--steps", type=int)
    p.add_argument("--sigmas", default=None, help="fixed schedule (ManualSigmas), replacing --scheduler / --steps")
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--sampler")
    p.add_argument("--scheduler")
    p.add_argument("--prompts", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sinks", action="store_true", help="profile SOL candidates with the model's sink rows exact")
    p.add_argument("--no-time-steps", dest="time_steps", action="store_false",
                   help="skip the plain generation that times a dense step")
    p.add_argument("--per-step", action="store_true", help="search per (layer, step) instead of per layer")
    p.add_argument("--reference-lora", default=None, help="existing config LoRA to compare against")
    p.add_argument("--targets", type=float, nargs="*", default=[3.0, 3.5, 4.0])
    p.add_argument("--weights", type=float, nargs="*", default=None,
                   help="weights of the secondary row groups (e.g. H3 audio) in the search proxy")
    p.add_argument("--no-trajectory", dest="trajectory", action="store_false",
                   help="verify: skip the whole-trajectory run per allocation (final-latent error, measured DiT time)")
    p.add_argument("--legacy-vram", action="store_true",
                   help="legacy estimate-based loading instead of DynamicVRAM (a ComfyUI server's default on NVIDIA)")
    p.add_argument("--only", default="", help="verify only these allocation names (comma-separated)")
    p.add_argument("--pick", default="ref_speed")
    adapter.add_args(p)
    args = p.parse_args()
    {"profile": profile, "search": search, "verify": verify, "write": write}[args.phase](adapter, args)
