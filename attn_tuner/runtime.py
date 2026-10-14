"""ComfyUI in-process: boot without a server, call nodes, and the helpers every phase shares."""

from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path


def boot_comfy(comfy_root: Path, extra_args: tuple[str, ...] = (), dynamic_vram: bool = False,
               vram_headroom_gb: float = 0.5):
    """Import ComfyUI in-process. ``dynamic_vram`` repeats main.py's DynamicVRAM (comfy-aimdo) set-up, which a server
    on NVIDIA uses by default: without it models load through the legacy estimate-based patcher, whose partial loads
    apply LoRAs to offloaded quantized layers in bf16 instead of requantizing them."""
    # The argument is not called `comfy`: `import comfy.options` below rebinds that name to the package.
    sys.argv = [sys.argv[0], "--disable-all-custom-nodes", *extra_args]
    sys.path.insert(0, str(comfy_root))
    import comfy.options

    comfy.options.enable_args_parsing()
    if dynamic_vram:
        import comfy_aimdo.control

        comfy_aimdo.control.init(simple_vram_headroom=None, nvml_pressure=True)
    import folder_paths  # noqa: F401
    from utils.extra_config import load_extra_path_config

    if (comfy_root / "extra_model_paths.yaml").exists():
        load_extra_path_config(str(comfy_root / "extra_model_paths.yaml"))
    import nodes

    if dynamic_vram:
        import comfy.memory_management
        import comfy.model_management
        import comfy.model_patcher
        import comfy_aimdo.control

        devices = comfy.model_management.get_all_torch_devices()
        if not comfy_aimdo.control.init_devices((d.index, int(vram_headroom_gb * 1024 ** 3)) for d in devices):
            raise RuntimeError("comfy-aimdo did not initialise; DynamicVRAM unavailable")
        comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
        comfy.memory_management.aimdo_enabled = True
    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
    return nodes.NODE_CLASS_MAPPINGS


def call(nodes_map, name: str, **inputs):
    cls = nodes_map[name]
    if hasattr(cls, "FUNCTION"):
        return getattr(cls(), cls.FUNCTION)(**inputs)
    out = cls.execute(**inputs)
    if inspect.iscoroutine(out):
        out = asyncio.run(out)
    return out.result if hasattr(out, "result") else out


def sample(nodes_map, model, cond, latent, args, seed: int):
    (noise,) = call(nodes_map, "RandomNoise", noise_seed=seed)
    (sampler,) = call(nodes_map, "KSamplerSelect", sampler_name=args.sampler)
    if getattr(args, "sigmas", None):  # distilled models ship a fixed schedule (ManualSigmas) instead of a scheduler
        (sigmas,) = call(nodes_map, "ManualSigmas", sigmas=args.sigmas)
    else:
        (sigmas,) = call(nodes_map, "BasicScheduler", model=model, scheduler=args.scheduler, steps=args.steps,
                         denoise=1.0)
    (guider,) = call(nodes_map, "BasicGuider", model=model, conditioning=cond)
    return call(nodes_map, "SamplerCustomAdvanced", noise=noise, guider=guider, sampler=sampler, sigmas=sigmas,
                latent_image=latent)


class StepOf:
    """Denoising step index of the current call, from the sigma in transformer_options (one host read per step)."""

    def __init__(self):
        self.tensor, self.index, self.schedule = None, None, None

    def __call__(self, kw) -> int | None:
        import torch

        to = kw.get("transformer_options", {})
        sigma, schedule = to.get("sigmas"), to.get("sample_sigmas")
        if sigma is None or schedule is None:
            return None
        if self.tensor is not sigma:
            sched = schedule.reshape(-1)[:-1].float().cpu()
            self.tensor = sigma
            self.index = int(torch.argmin((sched - float(sigma.reshape(-1)[0])).abs()))
            self.schedule = sched.tolist() + [float(schedule.reshape(-1)[-1])]
        return self.index


def rel(a, b) -> float:
    # float32 in chunks: a whole-tensor float copy of a 35K-row attention output does not fit next to the model
    a, b = a.reshape(-1), b.reshape(-1)
    num = den = 0.0
    for i in range(0, a.numel(), 1 << 26):
        x, y = a[i:i + (1 << 26)].float(), b[i:i + (1 << 26)].float()
        num += float(((x - y) ** 2).sum())
        den += float((y ** 2).sum())
    return (num / max(den, 1e-24)) ** 0.5
