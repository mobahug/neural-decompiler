"""Items 3, 4 (tensor digests), 5 and 6 against Experiment 022's local calibration table (read-only torch.load).

Own code throughout. Project code used only where the task allows it: ul (CUE_CLASSES, BIT, load_frozen_inputs) for
the frozen definitions and inputs, and b0c.exposed_units / b0c.draw_indices / b0c.draw_pairs in item 3 purely to
compare the calibration's index generation against this reviewer's own (no statistics over the 10,000 draws)."""

from __future__ import annotations

import hashlib
import math
import sys
import time

sys.path.insert(0, "/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/extractreview023")

import numpy as np
import torch

from revlib import CAND_DATA, CAND_INDEX, RECORD022, REPO, TABLE022, Report, strict_json, tensor_digest_own

T0 = time.time()
rep = Report("items 3, 4 (tensors), 5, 6")


def say(msg: str) -> None:
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


from neural_decompiler import upstream_localization as ul  # noqa: E402

index = strict_json(CAND_INDEX)
cells = np.frombuffer(CAND_DATA.read_bytes(), dtype="<f8").reshape(18900, 8)
COL = {name: i for i, name in enumerate(index["columns"])}
table = torch.load(TABLE022)
say(f"table loaded; keys {sorted(table)}")
rep.note("table dtypes: " + ", ".join(f"{k} {tuple(v.shape)} {v.dtype}" for k, v in table.items() if isinstance(v, torch.Tensor))
         + "; gates " + ", ".join(f"{k} {tuple(v.shape)} {v.dtype}" for k, v in sorted(table["gates"].items())))

# =====================================================================================================================
# Item 4: the 13 tensor digests, recomputed with this reviewer's own digest code.
record = strict_json(RECORD022)
bound = record["rematerialization"]["table_sha256"]
mine = {name: tensor_digest_own(table[name]) for name in ("dc", "dc_hat", "sse", "count", "mean", "m2", "ceiling", "historical")}
mine.update({f"gate_{name}": tensor_digest_own(values) for name, values in sorted(table["gates"].items())})
diff = sorted(k for k in set(mine) | set(bound) if mine.get(k) != bound.get(k))
rep.check("13 own tensor digests == 022 record's rematerialization.table_sha256", not diff and len(mine) == 13, f"differing {diff}")
rep.check("13 own tensor digests == index source_022.table_tensor_sha256", mine == index["source_022"]["table_tensor_sha256"])
# negative control: the digest is sensitive (one ulp in one dc element changes it)
probe = table["dc"].clone()
probe.view(-1)[12345] = float(np.nextafter(probe.view(-1)[12345].item(), np.inf))
rep.check("digest negative control (one-ulp change detected)", tensor_digest_own(probe) != mine["dc"])
del probe

# =====================================================================================================================
# Item 3: canonical ordering, independently.
CLASSES = tuple(ul.CUE_CLASSES)
rep.check("ul.CUE_CLASSES == (determiner-like, quantity, possessive-or-pronoun, adjective)", CLASSES == ("determiner-like", "quantity", "possessive-or-pronoun", "adjective"))
cues = index["cues"]
frames = index["frames"]
rep.check("index cues (175) == table's cues/token_ids/classes lists",
          len(cues) == 175 and [c[0] for c in cues] == list(table["cues"]) and [int(c[1]) for c in cues] == [int(t) for t in table["token_ids"]]
          and [c[2] for c in cues] == list(table["classes"]))
rep.check("index frames (108) == table's frames/templates lists",
          len(frames) == 108 and [f[0] for f in frames] == list(table["frames"]) and [f[1] for f in frames] == list(table["templates"]))
rank = [CLASSES.index(c[2]) for c in cues]
rep.check("classes appear in 022's class order (non-decreasing), all four present", rank == sorted(rank) and set(rank) == {0, 1, 2, 3})
ok_within = all(cues[i][1] < cues[i + 1][1] for i in range(len(cues) - 1) if cues[i][2] == cues[i + 1][2])
rep.check("token ids strictly ascending within each class", ok_within)
counts = {cls: sum(1 for c in cues if c[2] == cls) for cls in CLASSES}
rep.check("class counts 45/45/36/49", counts == {"determiner-like": 45, "quantity": 45, "possessive-or-pronoun": 36, "adjective": 49}, str(counts))
rep.check("cue token ids unique", len({c[1] for c in cues}) == 175)
fids = [f[0] for f in frames]
rep.check("frames strictly sorted by frame_id (unique)", fids == sorted(fids) and len(set(fids)) == 108)
rep.check("group == coordinated iff template == coordinated-adjective", all((f[2] == "coordinated") == (f[1] == "coordinated-adjective") and f[2] in ("coordinated", "cue_final") for f in frames))
n_coord = sum(1 for f in frames if f[2] == "coordinated")
rep.check("72 cue-final + 36 coordinated frames", n_coord == 36 and len(frames) - n_coord == 72, f"coordinated {n_coord}")
rep.check("P0/P1 masks == ul.BIT: emb|Bv|Bp = 14, +T = 30; index program masks match",
          ul.BIT["emb"] | ul.BIT["Bv"] | ul.BIT["Bp"] == 14 and (14 | ul.BIT["T"]) == 30 and ul.BIT == {"R": 1, "emb": 2, "Bv": 4, "Bp": 8, "T": 16}
          and index["program"]["P0_mask"] == 0 and index["program"]["P1_mask"] == {"cue_final": 14, "coordinated": 30})

inputs = ul.load_frozen_inputs(REPO)
say("frozen inputs loaded")
pool = inputs.pool
pool_nouns = [noun.lexical_key for noun in pool.nouns if noun.single_token]
rep.check("nouns (79) == table noun_keys == pool single-token nouns in order", index["nouns"] == list(table["noun_keys"]) == pool_nouns and len(pool_nouns) == 79)
# the pool frames, independently sorted
pool_frames = sorted(pool.frames, key=lambda frame: frame.frame_id)
rep.check("index frames == pool frames sorted by frame_id, templates equal",
          [f[0] for f in frames] == [fr.frame_id for fr in pool_frames] and [f[1] for f in frames] == [fr.template_id for fr in pool_frames])
# the pool cues, independently rebuilt from provenance (pool tokens of the four classes from the 011–019 cohorts, minus reference and plural cues)
excluded = set(int(v) for v in pool.reference_ids.values()) | {int(pool.token_id(name)) for name in pool.plural_cue.values()}
cohorts = {f"confirmation-0{k}" for k in range(11, 20)}
own_cues = []
for cls in CLASSES:
    members = sorted(((w, int(t)) for w, t in pool.tokens if pool.token_category[w] == cls and pool.token_source[w] in cohorts and int(t) not in excluded), key=lambda e: e[1])
    own_cues += [[w, t, cls] for w, t in members]
rep.check("index cues == own rebuild from the pool (classes in order, token id ascending)", own_cues == [[c[0], int(c[1]), c[2]] for c in cues], f"own {len(own_cues)}")
# Y2-like frames: first confirmed in 017–019, per template, by frame_id
origins = {"confirmation-017", "confirmation-018", "confirmation-019"}
own_y2 = {t: sorted(fr.frame_id for fr in pool.frames if fr.template_id == t and pool.frame_origin[fr.frame_id] in origins) for t in ("cardinal", "quantifier", "coordinated-adjective")}
rep.check("index y2_like_frames == own derivation (frame origin 017–019), 14 per template", index["y2_like_frames"] == own_y2 and all(len(v) == 14 for v in own_y2.values()),
          str({k: len(v) for k, v in own_y2.items()}))

# the tokenizer (local files only)
try:
    from transformers import AutoTokenizer

    from neural_decompiler.models import PYTHIA_70M

    tok = AutoTokenizer.from_pretrained(PYTHIA_70M.model_id, revision=PYTHIA_70M.revision, local_files_only=True)
    bad = [(w, t, tok.encode(" " + w, add_special_tokens=False)) for w, t, _ in cues if tok.encode(" " + w, add_special_tokens=False) != [int(t)]]
    rep.check(f"tokenizer ({PYTHIA_70M.model_id}@{PYTHIA_70M.revision[:7]}, local): ' '+word is exactly its single token id for all 175 cues", not bad, str(bad[:5]))
except Exception as error:  # noqa: BLE001
    rep.check("tokenizer check could run", False, repr(error))

# pronoun rows: present as provenance
pron = [i for i, c in enumerate(cues) if c[2] == "possessive-or-pronoun"]
pron_rows = [ci * 108 + fi for ci in pron for fi in range(108)]
rep.check("36 possessive-or-pronoun cues (3,888 pair rows) present, finite, n = 79",
          len(pron) == 36 and len(pron_rows) == 3888 and bool(np.isfinite(cells[pron_rows]).all()) and bool((cells[pron_rows, 0] == 79).all()),
          f"cue indices {pron[0]}–{pron[-1]}")

# the draws: own formula for all b in 0..9999, against b0c's index generation (indices only; no statistics)
from neural_decompiler import block0_completion as b0c  # noqa: E402

units = b0c.exposed_units(inputs)
rep.check("b0c.exposed_units order == index order (cues, frames)",
          [list(c) for c in units.cues] == [[c[0], int(c[1]), c[2]] for c in cues] and [fr.frame_id for fr in units.frames] == fids)
STRATA = ("determiner-like", "quantity", "adjective")
strata_rows = {s: [i for i, c in enumerate(cues) if c[2] == s] for s in STRATA}
y2_rows = {t: [fids.index(fid) for fid in index["y2_like_frames"][t]] for t in ("cardinal", "quantifier", "coordinated-adjective")}
sizes = {**{f"cue/{s}": len(strata_rows[s]) for s in STRATA}, **{f"frame/{t}": len(y2_rows[t]) for t in y2_rows}}
slots = {**{f"cue/{s}": 8 for s in STRATA}, **{f"frame/{t}": 6 for t in y2_rows}}
rep.note(f"stratum sizes {sizes}")
B = 10_000
say("computing own draw indices for 10,000 draws")
own_idx = {}
for stratum, n in sizes.items():
    arr = np.empty((B, slots[stratum]), dtype=np.int64)
    for b in range(B):
        for slot in range(slots[stratum]):
            arr[b, slot] = int.from_bytes(hashlib.sha256(f"023|primary|{b}|{stratum}|{slot}".encode()).digest()[:8], "big") % n
    own_idx[stratum] = arr
say("computing b0c.draw_indices(units, 10000)")
theirs = b0c.draw_indices(units, B)
rep.check("b0c.draw_indices keys == own strata (no pronoun stratum)", sorted(theirs) == sorted(own_idx), str(sorted(theirs)))
rep.check("own draw indices == b0c.draw_indices(units, 10000) for all 6 strata", all(np.array_equal(own_idx[s], theirs[s].numpy()) for s in own_idx))
# drawn cue rows never in the pronoun stratum; the pair index construction, all 10,000 draws, all four conditions
drawn_cues = np.concatenate([np.asarray(strata_rows[s])[own_idx[f"cue/{s}"]] for s in STRATA], axis=1)  # [B, 24]
rep.check("no drawn cue row belongs to the pronoun stratum (all 10,000 draws)", not np.isin(drawn_cues, pron).any())
group_frames = {"cue_final": [i for i, f in enumerate(frames) if f[2] == "cue_final"], "coordinated": [i for i, f in enumerate(frames) if f[2] == "coordinated"]}
for population in ("Y1", "Y2"):
    for group in ("cue_final", "coordinated"):
        if population == "Y1":
            fr = np.broadcast_to(np.asarray(group_frames[group]), (B, len(group_frames[group])))
        else:
            temps = ["coordinated-adjective"] if group == "coordinated" else ["cardinal", "quantifier"]
            fr = np.concatenate([np.asarray(y2_rows[t])[own_idx[f"frame/{t}"]] for t in temps], axis=1)
        own_pairs = (drawn_cues[:, :, None] * 108 + fr[:, None, :]).reshape(B, -1)
        their_pairs = b0c.draw_pairs(units, theirs, population, group, range(B)).numpy()
        cue_of = own_pairs // 108
        frame_of = own_pairs % 108
        ok_group = np.isin(frame_of, group_frames[group]).all()
        ok_y2 = True if population == "Y1" else np.isin(frame_of, sum(y2_rows.values(), [])).all()
        rep.check(f"{population}/{group}: own pair indices == b0c.draw_pairs for all 10,000 draws; no pronoun row; frames in group{' and Y2-like' if population == 'Y2' else ''}",
                  own_pairs.shape == their_pairs.shape and np.array_equal(own_pairs, their_pairs) and not np.isin(cue_of, pron).any() and ok_group and ok_y2,
                  f"shape {own_pairs.shape}")
        del own_pairs, their_pairs, cue_of, frame_of

# =====================================================================================================================
# Item 5: numerical reconstruction across all 18,900 rows.
dc = table["dc"].double()
dc_hat = table["dc_hat"].double()
ceiling = table["ceiling"].double()
mask_of_frame = torch.tensor([30 if f[1] == "coordinated-adjective" else 14 for f in frames])
p1 = torch.stack([dc_hat[:, fi, int(mask_of_frame[fi])] for fi in range(108)], dim=1)  # [175, 108, 79]
p0 = dc_hat[:, :, 0]
cf = [i for i, f in enumerate(frames) if f[2] == "cue_final"]
co = [i for i, f in enumerate(frames) if f[2] == "coordinated"]
rep.check("cue-final frames: stored dc_hat mask 30 == mask 14 bit for bit (T is a null player)", torch.equal(dc_hat[:, cf, 30], dc_hat[:, cf, 14]))
sse = table["sse"].double()


def compare(label: str, col: str, values: np.ndarray, expect_exact: bool = True) -> None:
    a = cells[:, COL[col]]
    d = np.abs(a - values)
    rel = d / np.maximum(np.abs(values), 1e-300)
    n_diff = int((a != values).sum())
    ok = n_diff == 0 if expect_exact else True
    rep.check(f"{label} {col}", ok, f"differing rows {n_diff}/18900, max |diff| {d.max():.3e}, max rel {rel.max():.3e}")


# (a) against the stored values
stored = {"n": table["count"].double().reshape(-1).numpy(), "SSE0": sse[:, :, 0].reshape(-1).numpy(),
          "SSE1": torch.stack([sse[:, fi, int(mask_of_frame[fi])] for fi in range(108)], dim=1).reshape(-1).numpy(),
          "mean": table["mean"].double().reshape(-1).numpy(), "M2": table["m2"].double().reshape(-1).numpy()}
for col, values in stored.items():
    compare("5a stored:", col, values)
# the mask choice really matters in coordinated frames (negative control): SSE1 is mask 30 there, not 14
s14 = sse[:, co, 14].reshape(-1).numpy()
s30 = sse[:, co, 30].reshape(-1).numpy()
coord_rows = np.array([ci * 108 + fi for ci in range(175) for fi in co])
rep.check("coordinated rows: artifact SSE1 == stored sse[mask 30] and != sse[mask 14] (mask-choice control)",
          np.array_equal(cells[coord_rows, COL["SSE1"]], sse[:, co, 30].reshape(-1).numpy()) and int((cells[coord_rows, COL["SSE1"]] != s14).sum()) > 0.99 * len(coord_rows),
          f"rows where mask 14 would differ: {int((s30 != s14).sum())}/{len(coord_rows)}")

# (b) own whole-table float64 reductions over the noun axis
own = {"S": dc.sum(-1), "Q": (dc * dc).sum(-1), "SSE0": ((dc - p0) ** 2).sum(-1), "SSE1": ((dc - p1) ** 2).sum(-1), "SSEC": ((dc - ceiling) ** 2).sum(-1)}
for col, values in own.items():
    compare("5b whole-table torch float64:", col, values.reshape(-1).numpy())
mean_t = dc.mean(-1, keepdim=True)
compare("5b whole-table torch float64 (descriptive):", "mean", mean_t.reshape(-1).numpy(), expect_exact=False)
compare("5b whole-table torch float64 (descriptive):", "M2", ((dc - mean_t) ** 2).sum(-1).reshape(-1).numpy(), expect_exact=False)

# (c) exactly-rounded sums (math.fsum over the float64 terms)
say("fsum reconstruction")
y_all = dc.reshape(18900, 79).numpy()
p0_all = p0.reshape(18900, 79).numpy()
p1_all = p1.reshape(18900, 79).numpy()
c_all = ceiling.reshape(18900, 79).numpy()
fs = {k: np.empty(18900) for k in ("S", "Q", "SSE0", "SSE1", "SSEC", "mean", "M2")}
for r in range(18900):
    y = y_all[r]
    s = math.fsum(y.tolist())
    fs["S"][r] = s
    fs["Q"][r] = math.fsum((y * y).tolist())
    fs["SSE0"][r] = math.fsum(((y - p0_all[r]) ** 2).tolist())
    fs["SSE1"][r] = math.fsum(((y - p1_all[r]) ** 2).tolist())
    fs["SSEC"][r] = math.fsum(((y - c_all[r]) ** 2).tolist())
    m = s / 79.0
    fs["mean"][r] = m
    fs["M2"][r] = math.fsum(((y - m) ** 2).tolist())
worst_rel = 0.0
for col, values in fs.items():
    a = cells[:, COL[col]]
    d = np.abs(a - values)
    rel = d / np.maximum(np.abs(values), 1e-300)
    worst_rel = max(worst_rel, float(rel.max()))
    rep.note(f"5c fsum vs artifact {col}: identical rows {int((a == values).sum())}/18900, max |diff| {d.max():.3e}, max rel {rel.max():.3e}")
rep.check("5c every column within 1e-13 relative of the exactly-rounded sums", worst_rel <= 1e-13, f"worst rel {worst_rel:.3e}")

# =====================================================================================================================
# Item 6: independent SST check on selections with repeated pairs.
say("item 6 selections")
n_col, s_col, q_col, mean_col, m2_col = (cells[:, COL[k]] for k in ("n", "S", "Q", "mean", "M2"))
rng = np.random.default_rng(2026)
selections = []
for k in range(400):
    size = int(rng.integers(100, 20001))
    selections.append((f"random-{k}-size{size}", rng.integers(0, 18900, size=size)))
all_rows = np.arange(18900)
cf_rows = np.array([ci * 108 + fi for ci in range(175) for fi in cf])
selections += [("all rows", all_rows), ("group cue_final", cf_rows), ("group coordinated", coord_rows),
               ("single row 0 x7", np.full(7, 0)), ("single row 12345 x7", np.full(7, 12345)),
               ("half the rows twice", np.concatenate([all_rows[:9450], all_rows[:9450]])),
               ("non-pronoun rows", np.array([r for r in all_rows if (r // 108) not in set(pron)])),
               ("one row x20000", np.full(20000, 777)), ("two rows alternating x5000", np.tile(np.array([3, 18899]), 5000))]
worst = {"onepass_vs_merge": (0.0, ""), "onepass_vs_direct": (0.0, ""), "merge_vs_direct": (0.0, ""), "onepass_fsum_vs_direct": (0.0, "")}
repeats = 0
for name, sel in selections:
    repeats += int(len(sel) - len(np.unique(sel)))
    N = n_col[sel].sum()
    S = s_col[sel].sum()
    Q = q_col[sel].sum()
    onepass = Q - S * S / N
    grand = S / N
    merge = m2_col[sel].sum() + (n_col[sel] * (mean_col[sel] - grand) ** 2).sum()
    Sf, Qf = math.fsum(s_col[sel].tolist()), math.fsum(q_col[sel].tolist())
    onepass_f = Qf - Sf * Sf / math.fsum(n_col[sel].tolist())
    y = y_all[sel].reshape(-1)
    my = math.fsum(y.tolist()) / y.size
    direct = math.fsum(((y - my) ** 2).tolist())
    for key, a, b in (("onepass_vs_merge", onepass, merge), ("onepass_vs_direct", onepass, direct), ("merge_vs_direct", merge, direct), ("onepass_fsum_vs_direct", onepass_f, direct)):
        rel = abs(a - b) / abs(b)
        if rel > worst[key][0]:
            worst[key] = (rel, name)
rep.note(f"item 6: {len(selections)} selections, {repeats} repeated pair uses in total")
for key, (rel, where) in worst.items():
    rep.check(f"item 6 SST {key}: max relative disagreement <= 1e-10", rel <= 1e-10, f"{rel:.3e} at {where}")
# illustrative: the pooled SST is not a sum of pair-local variances
sel = all_rows
rep.note(f"all rows: pooled SST {q_col.sum() - s_col.sum() ** 2 / n_col.sum():.6f} vs Σ M2 {m2_col.sum():.6f} (not equal, as the design requires)")
rep.summary()
say("done")
