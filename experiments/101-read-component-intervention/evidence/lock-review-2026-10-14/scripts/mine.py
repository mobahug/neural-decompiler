"""My own reconstructions for the lock review (no neural_decompiler import; numpy, torch tensors only as arithmetic).

Every function here is written from the design and plan text (and, where bitwise digests require it, from the
documented construction of 025's SHA-256 counter-mode Box-Muller stream), not by calling `ri`.
"""
from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

from common import ROOT, SNAPSHOT, canonical, content_digest, design_candidates, f64_digest, file_sha256, sha256_text

D_MODEL = 512
LOCK_011_PATH = ROOT / "experiments/011-encoding-read-prospective/preregistration-lock.json"
CONFIRMATION_PATH = ROOT / "experiments/101-read-component-intervention/confirmation-v1.json"
CALIBRATION_PATH = ROOT / "experiments/101-read-component-intervention/calibration-v1.json"
RESULTS_020 = ROOT / "outputs/experiment-020/results.json"
ISO_TAG = "101|iso|{frame_id}|{j}"


# ---------------------------------------------------------------------------
# The read covector.

def lock_011() -> dict:
    return json.loads(LOCK_011_PATH.read_text(encoding="utf-8"))


def wc_numpy(read_weight) -> tuple[np.ndarray, float]:
    """w_c = rw - mean(rw)·1 with a correctly rounded mean (math.fsum); float64."""
    rw = np.array(read_weight, dtype=np.float64)
    mean = math.fsum(float(x) for x in rw) / rw.size
    w_c = rw - mean
    return w_c, math.sqrt(math.fsum(float(x) * float(x) for x in w_c))


def wc_torch(read_weight):
    """The same formula with torch float64 ops (torch's mean and vector_norm), for a bitwise digest comparison."""
    import torch
    rw = torch.tensor(read_weight, dtype=torch.float64)
    w_c = rw - rw.mean()
    norm = float(torch.linalg.vector_norm(w_c))
    return w_c, norm, w_c / norm


def raw_weights() -> dict:
    from safetensors.numpy import load_file
    return load_file(str(SNAPSHOT / "model.safetensors"))


def read_weight_from_raw(weights: dict, d_T: np.ndarray, layer: int = 3, head: int = 4, n_heads: int = 8) -> dict:
    """γ₃ ⊙ (W_V @ (W_O @ d̂_T)) for head (layer, head), sliced by me from the GPT-NeoX checkpoint: the fused
    query_key_value rows are [n_heads, (q, k, v), d_head] along the output dim; dense's input columns are the heads."""
    qkv = weights[f"gpt_neox.layers.{layer}.attention.query_key_value.weight"].astype(np.float64)  # [3·d_model, d_model]
    dense = weights[f"gpt_neox.layers.{layer}.attention.dense.weight"].astype(np.float64)  # [d_model(out), d_model(in)]
    gamma = weights[f"gpt_neox.layers.{layer}.input_layernorm.weight"].astype(np.float64)
    d_model = dense.shape[0]
    d_head = d_model // n_heads
    rows = slice(head * 3 * d_head + 2 * d_head, head * 3 * d_head + 3 * d_head)
    W_V = qkv[rows, :].T  # [d_model, d_head]
    W_O = dense[:, head * d_head:(head + 1) * d_head].T  # [d_head, d_model]
    m = W_V @ (W_O @ d_T)
    rw = gamma * m
    return {"read_weight": rw, "W_V": W_V, "W_O": W_O, "gamma": gamma}


# ---------------------------------------------------------------------------
# The isotropic directions: SHA-256 counter-mode uniforms -> Box-Muller -> two Gram-Schmidt passes off (1/√d, ŵ) -> unit.

def sha_uniforms(tag: str, count: int) -> list[float]:
    values: list[float] = []
    counter = 0
    while len(values) < count:
        digest = hashlib.sha256(f"{tag}|{counter}".encode("utf-8")).digest()
        for k in range(4):
            top53 = int.from_bytes(digest[8 * k:8 * k + 8], "big") >> 11
            values.append((top53 + 1) / 2.0 ** 53)
        counter += 1
    return values[:count]


def gaussians(tag: str, count: int) -> list[float]:
    u = sha_uniforms(tag, 2 * ((count + 1) // 2))
    out: list[float] = []
    for i in range(0, len(u), 2):
        r = math.sqrt(-2.0 * math.log(u[i]))
        theta = 2.0 * math.pi * u[i + 1]
        out.append(r * math.cos(theta))
        out.append(r * math.sin(theta))
    return out[:count]


def iso_numpy(w_hat: np.ndarray, frame_id: str, j: int) -> np.ndarray:
    d = w_hat.size
    ones = np.full(d, 1.0 / math.sqrt(d))
    v = np.array(gaussians(ISO_TAG.format(frame_id=frame_id, j=j), d), dtype=np.float64)
    for _ in range(2):
        for b in (ones, w_hat):
            v = v - float(np.dot(v, b)) * b
    return v / float(np.linalg.norm(v))


def iso_torch(w_hat, frame_id: str, j: int):
    import torch
    d = int(w_hat.shape[0])
    ones = torch.full((d,), 1.0 / math.sqrt(d), dtype=torch.float64)
    v = torch.tensor(gaussians(ISO_TAG.format(frame_id=frame_id, j=j), d), dtype=torch.float64)
    for _ in range(2):
        for b in (ones, w_hat):
            v = v - torch.dot(v, b) * b
    return v / torch.linalg.vector_norm(v)


def tensor_digest_np(array: np.ndarray) -> str:
    return f64_digest(np.ascontiguousarray(array, dtype=np.float64).ravel().tolist(), list(array.shape))


def tensor_digest_torch(tensor) -> str:
    return f64_digest(tensor.detach().double().contiguous().reshape(-1).tolist(), list(tensor.shape))


# ---------------------------------------------------------------------------
# What the plan and design fix (written out by hand from the documents).

PRIMARY_ARMS = ["PL", "NOOP", "NEC-READ", "SUF-READ", "NEC-OFF", "SUF-OFF", "NEC-AMP", "SUF-NEG", "NEC-HALF", "SUF-HALF", "PC-SWAP-PL", "PC-SWAP-REF",
                *[f"NEC-ISO-{j}" for j in range(1, 9)], *[f"SUF-ISO-{j}" for j in range(1, 9)], "HEAD-NEC", "HEAD-SUF", "HEAD-NEC-V", "HEAD-SUF-V"]
STRATUM_ARMS = ["PL", "NEC-READ", "SUF-READ", "NEC-OFF", "SUF-OFF", "NEC-AMP", "SUF-NEG", "NEC-HALF", "SUF-HALF", "PC-SWAP-PL", "PC-SWAP-REF"]
REF_PROMPT_ARMS = {"REF", "SUF-READ", "SUF-OFF", "SUF-NEG", "SUF-HALF", "PC-SWAP-REF", "HEAD-SUF", "HEAD-SUF-V", *[f"SUF-ISO-{j}" for j in range(1, 9)]}
CUES = [("two", 767), ("three", 1264), ("many", 1142)]


def expected_tags() -> list[str]:
    return ["REF"] + [f"{arm}@two" for arm in PRIMARY_ARMS] + [f"{arm}@{cue}" for cue in ("three", "many") for arm in STRATUM_ARMS]


def binomial_tail(n: int, k: int) -> Fraction:
    return Fraction(sum(math.comb(n, i) for i in range(k, n + 1)), 2 ** n)


def holm_thresholds(n: int = 36, alpha: Fraction = Fraction(1, 40)) -> tuple[int, int, float, float]:
    better = next(k for k in range(n + 1) if binomial_tail(n, k) <= alpha / 2)
    other = next(k for k in range(n + 1) if binomial_tail(n, k) <= alpha)
    return better, other, float(binomial_tail(n, better)), float(binomial_tail(n, other))


def expected_configuration() -> dict:
    better, other, _, _ = holm_thresholds()
    tags = expected_tags()
    return {"name": "production", "n_frames": 36, "frame_candidates": design_candidates(), "expected_picks": list(range(1, 37)), "reference": ["ref", 581],
            "cues": [list(c) for c in CUES], "k_iso": 8, "n_scored_nouns": 79, "exposed_frames": 36, "holm_thresholds": [better, other], "d_threshold": 25,
            "p_threshold": 27, "precondition": 30, "tau_rank": math.ceil(Fraction(90, 100) * 36), "max_overshoot_frames": 6, "max_weak_frames": 6,
            "runs_per_frame": len(tags), "tags": tags}


EXPECTED_TOLERANCES = {"G1_read_weight": 1e-9, "G1_basis": 1e-12, "G2_read": 1e-4, "G2_length": 1e-4, "G2_direction": 1e-4, "G2_u_off32": 1e-6,
                       "G3_pass_through": 1e-4, "G4_I3": 1e-4, "G4_I4": 1e-3, "G4_level1": 2e-2, "G4_readout": 2e-2, "G4_logit": 2e-2, "G5_row": 1e-4,
                       "G5_output": 1e-3, "environment": 1e-9}

DESIGN_OUTCOME_TABLE = [  # design §7, in order (the first row extended by the plan: an incident also renders NOT_INTERPRETABLE)
    ("any gate in §8 fails", "NOT_INTERPRETABLE"), ("the precondition fails", "BEHAVIOR_PRECONDITION_NOT_MET"),
    ("neither R nor I passes", "READ_COMPONENT_EFFECTS_NOT_ESTABLISHED"), ("R passes, I does not", "READ_REMOVAL_EFFECT_INSTALLATION_NOT_ESTABLISHED"),
    ("I passes, R does not", "READ_INSTALLATION_EFFECT_REMOVAL_NOT_ESTABLISHED"), ("R and I pass, D fails", "READ_COMPONENT_EFFECTS_DIRECTION_NOT_ESTABLISHED"),
    ("R, I and D pass, P is uninformative", "READ_COMPONENT_EFFECTS_PROGRAM_NOT_EVALUABLE"), ("R, I and D pass, P fails", "READ_COMPONENT_EFFECTS_PROGRAM_NOT_CONFIRMED"),
    ("R, I, D and P pass", "READ_COMPONENT_EFFECTS_AS_PREDICTED")]


# ---------------------------------------------------------------------------
# The confirmation file, the manifests and the stage-1 layout.

def confirmation() -> dict:
    return json.loads(CONFIRMATION_PATH.read_text(encoding="utf-8"))


def my_manifest(frames: list[dict]) -> dict:
    s1, s2 = [], []
    prompt = {"ref": ("ref", 581), "two": ("two", 767), "three": ("three", 1264), "many": ("many", 1142)}
    for frame in frames:
        fid = frame["frame_id"]
        s1.append(f"{fid}|ref|581|S1")
        for tag in expected_tags():
            arm, _, cue = tag.partition("@")
            label, token = prompt["ref"] if arm in REF_PROMPT_ARMS else prompt[cue]
            s2.append(f"{fid}|{label}|{token}|{tag}")
    return {"S1-REF": sorted(s1), "S2-TARGET": sorted(s2), "_s2_count": len(s2), "_s2_distinct": len(set(s2)),
            "_untagged": len({"|".join(k.split("|")[:3]) for k in s1 + s2})}


def my_layout(frames: list[dict], d_model: int, n_heads: int, n_nouns: int) -> list[dict]:
    out = []
    for frame in frames:
        keys = int(frame["p_t"]) + 1
        shapes = [("x1_all", [keys, d_model]), ("x2_all", [keys, d_model]), ("x3_all", [keys, d_model]), ("x4_all", [keys, d_model]), ("x5_all", [keys, d_model]),
                  ("h6", [d_model]), ("rows4", [n_heads, keys]), ("rows5", [n_heads, keys]), ("c_ref", [n_nouns])]
        out += [{"name": f"{frame['frame_id']}/{name}", "shape": shape} for name, shape in shapes]
    return out


def locked_entry_shapes(entry: dict) -> dict:
    def shape(value):
        dims = []
        while isinstance(value, list):
            dims.append(len(value))
            value = value[0] if value else None
        return dims
    return {name: shape(value) for name, value in entry.items() if isinstance(value, list)}


# ---------------------------------------------------------------------------
# The nouns: 020's pool order (the 60 manifest nouns by split, first appearance; then 006's 20), scorable = single tokens.

def noun_order() -> tuple[list[str], list[str]]:
    manifest = json.loads((ROOT / "screening/behavior-candidates/manifest-v1.json").read_text(encoding="utf-8"))
    nouns: list[tuple[str, bool]] = []
    for split in ("selection-development", "selection-holdout", "future-reserve"):
        seen: dict[str, bool] = {}
        for case in manifest["cases"]:
            if case["candidate_id"] == "regular-plural" and case["split"] == split and case["lexical_key"] not in seen:
                xa = case["conditions"]["x_a"]
                seen[case["lexical_key"]] = len(xa["a_token_ids"]) == 1 and len(xa["b_token_ids"]) == 1
        if len(seen) != 20:
            raise SystemExit(f"{split}: {len(seen)} nouns")
        nouns += list(seen.items())
    c006 = json.loads((ROOT / "experiments/006-low-rank-cue-decompilation/confirmation-v1.json").read_text(encoding="utf-8"))
    nouns += [(entry["lexical_key"], True) for entry in c006["nouns"]]
    return [key for key, _ in nouns], [key for key, scorable in nouns if scorable]
