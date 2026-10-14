"""What the tuner needs to know about a model; one subclass per architecture."""

from __future__ import annotations

import argparse


class Adapter:
    name = ""  # CLI name and LoRA file prefix
    title = ""  # human-readable, for LoRA metadata
    # Row groups whose attention-output error is tracked separately (the first is the primary one). The model's output
    # is split the same way in verify.
    groups: tuple[str, ...] = ()
    # SOL options for the "s" (sink) candidates: names of row spans the model publishes in
    # transformer_options["attention_spans"] (see `spans`)
    sinks: dict = {}
    calib: list[str] = []  # profile prompts
    eval: dict[str, str] = {}  # verify prompts
    min_tokens = 4096  # shorter attention calls (text refiners and the like) are left dense

    def add_args(self, p: argparse.ArgumentParser):
        """Model-specific flags and defaults for the shared ones (unet, steps, width, height, sampler, scheduler)."""

    def load(self, nodes_map, args):
        """(model patcher with the stack applied, context for `conditioning`); loaded once per run."""
        raise NotImplementedError

    def conditioning(self, nodes_map, ctx, args, prompt: str):
        """(positive conditioning, latent) for one prompt."""
        raise NotImplementedError

    def rows(self, to: dict) -> dict[str, list[tuple[int, int]]] | None:
        """Row spans per group for a DiT self-attention call, or None to leave the call alone."""
        raise NotImplementedError

    def spans(self, to: dict) -> dict:
        """transformer_options for the candidates: sink spans the model may not publish itself yet."""
        return to

    def split_output(self, x, c) -> dict:
        """Model output per group, for the one-step error in verify."""
        raise NotImplementedError

    def split_latent(self, samples) -> dict:
        """Sampler output per group, for the final-latent error in verify; one group by default."""
        return {self.groups[0]: samples}

    def config_key(self, layer: str) -> str:
        return f"diffusion_model.blocks.{layer}.attn.comfy_attention.config"
