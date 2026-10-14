"""CPU tests for the search, the LoRA format and the helpers; no ComfyUI or GPU needed.

    <comfy venv>/bin/python -m pytest tests
"""

import itertools
import json
import random

import pytest
import torch
from safetensors.torch import save_file

from attn_tuner.models import ADAPTERS
from attn_tuner.runtime import rel
from attn_tuner.tuner import cost, frontier, layer_configs, load_reference, public_stack, spearman

GROUPS = ("video", "audio")


def random_items(n_layers, cands, seed):
    rng = random.Random(seed)
    items = {}
    for l in range(n_layers):
        row = {"dense": {"ms": 20.0, "err": 0.0, "err_video": 0.0, "err_audio": 0.0}}
        for c in cands:
            # times on the 2 ms grid, so the knapsack's rounding is exact and brute force is a fair reference
            v, a = rng.uniform(0.01, 0.3), rng.uniform(0.01, 0.3)
            row[c] = {"ms": 2.0 * rng.randint(1, 9), "err": (v + a) / 2, "err_video": v, "err_audio": a}
        items[str(l)] = row
    return items


def brute_force(items, weight, budget):
    keys = sorted(items)
    best = None
    for picks in itertools.product(*(list(items[k]) for k in keys)):
        alloc = dict(zip(keys, picks))
        t, _ = cost(items, alloc, GROUPS)
        if t <= budget + 1e-9:
            e = cost(items, alloc, GROUPS, weight)[1]
            best = e if best is None else min(best, e)
    return best


@pytest.mark.parametrize("weight", [None, 1.0, 10.0])
@pytest.mark.parametrize("seed", range(3))
def test_frontier_is_exact(weight, seed):
    items = random_items(5, ["int8", "sol1"], seed)
    front = frontier(items, GROUPS, weight)
    assert front, "empty frontier"
    for t, e, alloc in front:
        full = {k: alloc.get(k, "dense") for k in items}
        # the reported point is achievable and matches its own allocation
        assert cost(items, full, GROUPS)[0] == pytest.approx(t)
        assert cost(items, full, GROUPS, weight)[1] == pytest.approx(e)
        # and nothing at that budget does better
        assert e == pytest.approx(brute_force(items, weight, t))
    times = [t for t, _, _ in front]
    errs = [e for _, e, _ in front]
    assert times == sorted(times) and errs == sorted(errs, reverse=True), "frontier must be Pareto-ordered"


def test_spearman():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert abs(spearman([1, 2, 3, 4, 5], [2, 5, 1, 4, 3])) < 1.0


def test_layer_configs_per_layer_and_per_step():
    cands = {"dense": None, "int8": {"attention": "comfy_kitchen_int8"},
             "sol1": {"attention": "comfy_kitchen_sol", "tau": 1.0}}
    schedule = [1.0, 0.8, 0.5, 0.0]
    assert layer_configs({"0": "int8", "3": "sol1"}, cands, schedule) == {"0": cands["int8"], "3": cands["sol1"]}
    # per step: one kernel at every step collapses to a plain config; all-dense layers are left out
    out = layer_configs({"0:0": "sol1", "0:1": "int8", "0:2": "int8", "1:0": "int8", "1:1": "int8", "1:2": "int8"},
                        cands, schedule)
    assert out["1"] == cands["int8"]
    assert "2" not in out
    ranges = [c["sigma_range"] for c in out["0"]]
    # each step's range is split halfway between its sigma and the neighbours', covering the whole schedule
    assert ranges == [[0.9, 1e9], [0.65, 0.9], [0.0, 0.65]]
    assert [c["attention"] for c in out["0"]] == ["comfy_kitchen_sol", "comfy_kitchen_int8", "comfy_kitchen_int8"]


def test_load_reference_round_trip(tmp_path):
    adapter = ADAPTERS["minimax_h3"]
    configs = {"0": {"attention": "comfy_kitchen_int8"},
               "1": {"attention": "comfy_kitchen_sol", "tau": 0.5, **adapter.sinks},
               "2": {"attention": "comfy_kitchen_sol", "tau": 1.3}}
    tensors = {adapter.config_key(l): torch.tensor(list(json.dumps(c).encode()), dtype=torch.uint8)
               for l, c in configs.items()}
    # H3's text refiner has its own blocks.N: it must not be read as DiT block N
    tensors["diffusion_model.token_refiner.blocks.0.attn.comfy_attention.config"] = torch.tensor(
        list(json.dumps({"attention": "comfy_kitchen_sol", "tau": 2.0}).encode()), dtype=torch.uint8)
    path = tmp_path / "attn.safetensors"
    save_file(tensors, str(path))
    assert load_reference(adapter, str(path)) == {"0": "int8", "1": "sol0.5s", "2": "sol1.3"}


def test_public_stack_drops_paths():
    meta = {"unet": "u.safetensors", "candidates": {"x": 1}, "ref_image": ["/home/a/1.png", "/home/a/2.png"]}
    assert public_stack(meta) == {"unet": "u.safetensors", "ref_images": 2}
    assert public_stack({"ref_image": "/home/a/1.png"})["ref_images"] == 1  # older profiles stored one str
    assert "ref_images" not in public_stack({"ref_image": []})


def test_rel_matches_direct_and_chunks():
    g = torch.Generator().manual_seed(0)
    b = torch.randn(3, (1 << 26) // 3 + 7, generator=g)  # crosses a chunk boundary
    a = b + 0.01 * torch.randn(b.shape, generator=g)
    direct = float((a.double() - b.double()).norm() / b.double().norm())  # float32 norms drift at this size
    assert rel(a, b) == pytest.approx(direct, rel=1e-5)
    assert rel(a.bfloat16(), b.bfloat16()) == pytest.approx(
        float((a.bfloat16().double() - b.bfloat16().double()).norm() / b.bfloat16().double().norm()), rel=1e-5)
    assert rel(b, b) == 0.0
