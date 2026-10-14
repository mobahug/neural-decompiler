"""Items 3-6 and 8, my own code first (no neural_decompiler import): the lock's content field by field against what I
derive from the committed files, the raw checkpoint and the documents."""
from common import install_fs_guard

install_fs_guard()

import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import mine  # noqa: E402
from common import EVENTS, ROOT, SNAPSHOT, canonical, content_digest, design_candidates, f64_digest, file_sha256, forbidden_101_sequences, git, \
    git_bytes, git_blob, sha256_text, tokenizer, write_json  # noqa: E402

EXP = "experiments/101-read-component-intervention"
OUT: dict = {}
lock_bytes = (ROOT / "outputs/experiment-101/candidate-lock.json").read_bytes()
lock = json.loads(lock_bytes)
prereg_bytes = (ROOT / "outputs/experiment-101/candidate-preregistration.md").read_bytes()
state = json.loads((ROOT / "outputs/experiment-101/results.json").read_bytes())
problems: list[str] = []


def check(name: str, ok: bool, detail=None):
    OUT.setdefault("checks", {})[name] = {"ok": bool(ok), **({"detail": detail} if detail is not None else {})}
    if not ok:
        problems.append(name)


# ---------------------------------------------------------------------------
# 1. The lock's own digest and form.
check("lock_file_sha256", hashlib.sha256(lock_bytes).hexdigest() == "8dc6e6d38c876a2dd9bf905fbeaf36b774b487967597795beb032bb4a0751087")
check("lock_content_sha256_recomputes", content_digest(lock) == lock["content_sha256"] == "8da97e35556d0baa4739899cb841e86cbd59cced21611c732fdd9d8c9edd9061",
      content_digest(lock))
check("lock_file_is_canonical_plus_newline", lock_bytes.decode("utf-8") == canonical(lock) + "\n")
check("lock_top_level_keys", sorted(lock) == sorted(["calibration_101", "configuration", "confirmation_101", "constants", "content_sha256", "cue_order", "dependencies",
                                                     "design", "experiment", "geometry", "inherited_025", "inputs", "kind", "module", "module_blobs", "noun_keys",
                                                     "outcome", "plan", "protocol_code_commit", "run_id", "run_order", "schema_version", "semantics", "stage1",
                                                     "statistics", "tolerances"]), sorted(lock))
check("prereg_file_sha256", hashlib.sha256(prereg_bytes).hexdigest() == "3761b8130203d5d15ac0e30055aa135626a6d97f0ab807df6cf4d31b6af8e980")
check("experiment_schema_kind", (lock["experiment"], lock["schema_version"], lock["kind"]) == ("101", 1, "the Experiment 101 preregistration lock"))

# 2. run id, commit, design, plan.
check("run_id_is_state_run_id", lock["run_id"] == state["run_id"] == "30ca8697c8702496")
head = git("rev-parse", "HEAD").strip()
check("protocol_code_commit_is_head_and_lock_phase_commit", lock["protocol_code_commit"] == head == state["phases"]["lock"]["commit"])
check("design_record", lock["design"] == {"commit": "a87e284", "decision": "516df20", "path": "docs/superpowers/specs/2026-10-01-experiment-101-read-component-intervention-design.md",
                                          "revision": 3})
check("plan_record", lock["plan"] == {"commit": "96f0cc8", "decision": "1e48c1c", "path": "docs/superpowers/plans/2026-10-05-experiment-101-read-component-intervention-plan.md",
                                      "revision": 3})
for key in ("design", "plan"):
    for field in ("commit", "decision"):
        sha = lock[key][field]
        check(f"{key}_{field}_touches_the_file", sha in git("log", "--format=%h", "--abbrev=7", "--", lock[key]["path"]).split(), sha)

# 3. The configuration against the documents.
expected_config = mine.expected_configuration()
diff = sorted(k for k in set(expected_config) | set(lock["configuration"]) if expected_config.get(k) != lock["configuration"].get(k))
check("configuration_equals_plan_and_design", not diff, diff)
check("run_order_equals_plan", lock["run_order"] == mine.expected_tags() and len(lock["run_order"]) == 55)
check("cue_order_equals_plan", lock["cue_order"] == ["two", "three", "many"])
better, other, tail_b, tail_o = mine.holm_thresholds()
OUT["holm"] = {"thresholds": [better, other], "tails": [tail_b, tail_o]}
check("holm_26_25_tails", (better, other) == (26, 25) and abs(tail_b - 0.005665) < 1e-6 and abs(tail_o - 0.014408) < 1e-6, OUT["holm"])

# 4. Module blobs: the lock's 20 against git ls-tree at HEAD and at 516df20; the module's own blob.
def ls_tree(rev):
    rows = {}
    for line in git("ls-tree", rev, "src/neural_decompiler/").splitlines():
        meta, path = line.split("\t")
        rows[path.rsplit("/", 1)[1]] = meta.split()[2]
    return rows
head_tree, design_tree = ls_tree("HEAD"), ls_tree("516df20")
check("module_blobs_20", len(lock["module_blobs"]) == 20)
check("module_blobs_equal_git_head", all(head_tree.get(name) == blob for name, blob in lock["module_blobs"].items()),
      {n: (b, head_tree.get(n)) for n, b in lock["module_blobs"].items() if head_tree.get(n) != b})
check("module_blobs_equal_git_516df20", all(design_tree.get(name) == blob for name, blob in lock["module_blobs"].items()))
check("module_blobs_equal_working_files", all(git_blob((ROOT / "src/neural_decompiler" / name).read_bytes()) == blob for name, blob in lock["module_blobs"].items()))
check("dependencies_module_blobs_equal_lock_module_blobs", lock["dependencies"]["module_blobs"] == lock["module_blobs"])
own = git_blob((ROOT / "src/neural_decompiler/read_intervention.py").read_bytes())
check("own_module_blob", lock["module"] == {"path": "src/neural_decompiler/read_intervention.py", "blob": own} and head_tree["read_intervention.py"] == own, own)
check("plan_blob_table_equals_lock", True)  # filled below from the plan text
plan_text = (ROOT / lock["plan"]["path"]).read_text(encoding="utf-8")
plan_blobs = {}
for line in plan_text.splitlines():
    s = line.strip()
    if s.startswith("| `") and s.count("|") == 3:
        name, blob = [c.strip().strip("`") for c in s.strip("|").split("|")]
        if len(blob) == 40:
            plan_blobs[f"{name}.py"] = blob
check("plan_blob_table_equals_lock", plan_blobs == lock["module_blobs"], sorted(set(plan_blobs.items()) ^ set(lock["module_blobs"].items())))

# 5. The 025 inheritance.
p025 = ROOT / lock["inherited_025"]["path"]
c025 = json.loads(p025.read_text(encoding="utf-8"))
check("inherited_025", file_sha256(p025) == lock["inherited_025"]["confirmation_file_sha256"] and content_digest(c025) == c025["content_sha256"]
      == lock["inherited_025"]["confirmation_content_sha256"] and lock["inputs"]["025_confirmation_file"] == file_sha256(p025)
      and lock["inputs"]["025_confirmation_content"] == c025["content_sha256"])

# 6. The confirmation binding.
conf_path = ROOT / f"{EXP}/confirmation-v1.json"
conf = json.loads(conf_path.read_text(encoding="utf-8"))
my_manifest = mine.my_manifest(conf["frames"])
binding = lock["confirmation_101"]
check("confirmation_file_sha256", binding["file_sha256"] == file_sha256(conf_path))
check("confirmation_content_sha256", binding["content_sha256"] == content_digest(conf) == conf["content_sha256"])
check("confirmation_manifest_equals_mine", conf["manifest"] == {"S1-REF": my_manifest["S1-REF"], "S2-TARGET": my_manifest["S2-TARGET"]})
check("confirmation_manifest_sha256", binding["manifest_sha256"] == sha256_text(canonical(conf["manifest"])) == conf["manifest_sha256"])
check("confirmation_counts", binding["counts"] == {"S1-REF": 36, "S2-TARGET": 1980, "frames": 36, "runs_per_frame": 55, "untagged_prompts": 144}
      and my_manifest["_s2_count"] == my_manifest["_s2_distinct"] == 1980 and my_manifest["_untagged"] == 144 and binding["path"] == f"{EXP}/confirmation-v1.json")
check("confirmation_s1_s2_disjoint", not set(my_manifest["S1-REF"]) & set(my_manifest["S2-TARGET"]))
check("confirmation_frames_are_candidates_1_36", [f["frame_id"] for f in conf["frames"]] == [f"coordinated-adjective-101-{n:02d}" for n in range(1, 37)]
      and [f["text_template"] for f in conf["frames"]] == design_candidates()[:36])
# the frozen frames' token ids against my tokenization of the design texts with `one`
tok = tokenizer()
mismatch = []
for f, text in zip(conf["frames"], design_candidates()[:36]):
    ids = tok.encode(text.format(cue="one"), add_special_tokens=False).ids
    if ids != f["prefix_ids"] + [581] + f["suffix_ids"] or f["p_c"] != len(f["prefix_ids"]) or f["p_t"] != f["p_c"] + 1 or len(f["suffix_ids"]) != 1:
        mismatch.append(f["frame_id"])
check("confirmation_frames_tokenize_from_design", not mismatch, mismatch)
check("state_confirmation_binding_equals_lock", state["confirmation_101"] == binding)

# 7. The calibration binding; τ from the calibration's own sorted error lists.
cal_path = ROOT / f"{EXP}/calibration-v1.json"
cal = json.loads(cal_path.read_text(encoding="utf-8"))
cb = lock["calibration_101"]
check("calibration_binding", cb == {"path": f"{EXP}/calibration-v1.json", "file_sha256": file_sha256(cal_path), "content_sha256": content_digest(cal),
                                    "tau_rem": cal["tau"]["rem"], "tau_ins": cal["tau"]["ins"], "evaluable": cal["tau"]["evaluable"]})
check("calibration_content_recomputes", content_digest(cal) == cal["content_sha256"] == "1eb1fe68ac2d49228ff8eff6c72391cc0695eaeeb3ae82ed122f9aa738f42e1e")
err_rem, err_ins = cal["distributions"]["P_error_rem"], cal["distributions"]["P_error_ins"]
OUT["tau_recompute"] = {"n_rem": len(err_rem), "n_ins": len(err_ins), "rank33_rem": sorted(err_rem)[32], "rank33_ins": sorted(err_ins)[32],
                        "nearest_rank": math.ceil(0.9 * 36)}
check("tau_is_33rd_of_36", len(err_rem) == len(err_ins) == 36 and sorted(err_rem)[32] == cb["tau_rem"] and sorted(err_ins)[32] == cb["tau_ins"], OUT["tau_recompute"])
check("tau_evaluable", cb["evaluable"] is True and cb["tau_rem"] <= 0.20 and cb["tau_ins"] <= 0.20)
check("calibration_no_trigger", not any(entry["fired"] for entry in cal["triggers"].values()))
check("calibration_confirmation_binding", cal["confirmation_101"] == binding)
check("calibration_module_blob_is_head", cal["module_blob"] == own)
check("calibration_tolerances_equal_lock", cal["tolerances"] == lock["tolerances"])
# what from the calibration enters the lock: exactly the binding (no shares, power, distribution, report)
lock_text = canonical(lock)
leaks = [name for name in ("shares", "power", "distributions", "reports", "min_s", "measurements_sha256") if f'"{name}"' in lock_text]
cal_numbers = {repr(v) for v in cal["shares"].values()} | {repr(cal["power"]["P"])}
check("lock_binds_only_tau_from_calibration", not leaks, leaks)

# 8. Dependencies: 020's readout, the exposed states digest, the model digests (embedding from the raw checkpoint).
res020_raw = (ROOT / "outputs/experiment-020/results.json").read_bytes()
res020 = json.loads(res020_raw)
locked = res020["exploration"]["locked_states"]
exposed_digest = sha256_text(canonical({fid: sha256_text(canonical(entry)) for fid, entry in sorted(locked.items())}))
dep = lock["dependencies"]
check("readout_020_blob", dep["readout_020"]["blob"] == head_tree["readout_decompilation.py"] == lock["module_blobs"]["readout_decompilation.py"])
check("exposed_states_sha256", dep["readout_020"]["exposed_states_sha256"] == exposed_digest, exposed_digest)
check("results_020_file", lock["inputs"]["results_020_file"] == hashlib.sha256(res020_raw).hexdigest() == "da63b8c29f9553a9da62bea7a442e11cef999ccddc71a7a332ac7c938abc9e00")
rec020 = res020.pop("state_sha256")
check("results_020_state", lock["inputs"]["results_020_state"] == rec020 == sha256_text(canonical(res020)))
check("readout_020_frozen_input_digests_subset", all(lock["inputs"][k] == v for k, v in dep["readout_020"]["frozen_input_digests"].items())
      and len(dep["readout_020"]["frozen_input_digests"]) == 21)
weights = mine.raw_weights()
emb = weights["gpt_neox.embed_in.weight"]
emb_digest = hashlib.sha256(json.dumps(list(emb.shape)).encode("ascii") + b"|<f4|" + np.ascontiguousarray(emb, dtype="<f4").tobytes()).hexdigest()
check("embedding_sha256_from_raw_checkpoint", dep["model"]["embedding_sha256"] == emb_digest, emb_digest)
check("model_id_revision", (dep["model"]["model_id"], dep["model"]["revision"]) == ("EleutherAI/pythia-70m-deduped", "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c"))
check("checkpoint_file_sha256", file_sha256(SNAPSHOT / "model.safetensors") == "3da388330e4549156d76b58d6d268c63cd005e9336b4f4d2d378421e7b7a33fd")
# the parameters digest by HF names (it equals the lock's only if the bridge exposes the HF names; recorded, not required)
h = hashlib.sha256()
for name in sorted(weights):
    arr = np.ascontiguousarray(weights[name], dtype="<f4")
    h.update(name.encode("utf-8") + b"|" + json.dumps(list(arr.shape)).encode("ascii") + b"|" + arr.tobytes())
OUT["parameters_digest_by_hf_names"] = {"digest": h.hexdigest(), "equals_lock": h.hexdigest() == dep["model"]["parameters_sha256"], "n_tensors": len(weights)}
check("dependencies_text_fields_present", isinstance(dep["measurement"], str) and dep["table"] == lock["stage1"]["table"]["construction"])

# 9. Every input digest that names a committed file: match it against the files of 020, 022-025 (and 011/012/017 locks).
files = {}
for d in ("experiments/020-readout-decompilation", "experiments/022-upstream-error-localization", "experiments/023-block0-completion", "experiments/024-readout-routing-nounness",
          "experiments/025-nounness-direction-intervention"):
    for p in (ROOT / d).rglob("*"):
        if p.is_file() and "/evidence/" not in str(p) and "__pycache__" not in str(p):
            files[str(p.relative_to(ROOT))] = p
by_sha = {}
contents = {}
for rel, p in files.items():
    by_sha.setdefault(file_sha256(p), []).append(rel)
    if p.suffix == ".json":
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict) and "content_sha256" in payload:
            contents[rel] = (payload["content_sha256"], content_digest(payload))
input_match = {}
for key, value in lock["inputs"].items():
    if key.endswith("_file"):
        input_match[key] = by_sha.get(value, ["<no committed file>"] if key != "results_020_file" else ["outputs/experiment-020/results.json"])
    elif key.endswith("_content"):
        hits = [rel for rel, (recorded, recomputed) in contents.items() if recorded == value == recomputed]
        input_match[key] = hits or ["<no committed file with this verified content digest>"]
for key, rel in (("lock_011", "experiments/011-encoding-read-prospective/preregistration-lock.json"), ("lock_012", "experiments/012-layer-correction-token-local/preregistration-lock.json"),
                 ("lock_017", "experiments/017-transport-head-pattern/preregistration-lock.json"), ("closure_020", "experiments/020-readout-decompilation/closure.json"),
                 ("extract_020", "experiments/020-readout-decompilation/evidence/exploration-record-2026-09-22.json"),
                 ("manifest", "screening/behavior-candidates/manifest-v1.json"), ("extension", "experiments/005-regular-plural-mechanism/extension-v1.json")):
    payload = json.loads((ROOT / rel).read_text(encoding="utf-8"))
    input_match[key] = {"path": rel, "recorded": payload.get("content_sha256"), "recomputed": content_digest(payload), "lock": lock["inputs"][key]}
OUT["input_match"] = input_match
unmatched = [k for k, v in input_match.items() if (isinstance(v, list) and v and v[0].startswith("<")) or (isinstance(v, dict) and not (v["recorded"] == v["lock"]))]
check("input_digests_traced_to_committed_files", not unmatched, unmatched)
check("inputs_count_42", len(lock["inputs"]) == 42 and state["inputs"] == lock["inputs"], len(lock["inputs"]))

# 10. The nouns: 020's pool order rebuilt from the manifest and 006's file; scorable = single-token forms.
all_nouns, scorable = mine.noun_order()
check("noun_keys_79_in_pool_order", lock["noun_keys"] == scorable and len(scorable) == 79 and len(all_nouns) == 80 and set(all_nouns) - set(scorable) == {"peach"},
      {"n_all": len(all_nouns), "n_scorable": len(scorable), "dropped": sorted(set(all_nouns) - set(scorable))})
check("noun_keys_in_stage1_table_meta", lock["stage1"]["table"]["noun_keys"] == lock["noun_keys"])
ex_nouns = res020["exploration"]["nouns"]
check("noun_keys_equal_020_scorable_set", set(lock["noun_keys"]) == {k for k, v in ex_nouns.items() if v["scorable"]})

# 11. The geometry: w_c from 011's lock (mine, numpy fsum and torch), from the raw weights; digests; ‖w_c‖.
import torch  # noqa: E402  (arithmetic only; no model)
l011 = mine.lock_011()
d_T = np.array(l011["axes_vectors"]["T"], dtype=np.float64)
wc_np, norm_np = mine.wc_numpy(l011["read_weight"])
wc_t, norm_t, what_t = mine.wc_torch(l011["read_weight"])
geo = lock["geometry"]
OUT["w_c"] = {"norm_numpy_fsum": norm_np, "norm_torch": norm_t, "norm_lock": geo["w_c_norm"], "digest_numpy": mine.tensor_digest_np(wc_np),
              "digest_torch": mine.tensor_digest_torch(wc_t), "digest_lock": geo["w_c_sha256"], "max_abs_numpy_vs_torch": float(np.max(np.abs(wc_np - wc_t.numpy()))),
              "w_hat_digest_torch": mine.tensor_digest_torch(what_t), "w_hat_digest_lock": geo["w_hat_sha256"],
              "d_T_digest": mine.tensor_digest_np(d_T), "d_T_digest_lock": geo["d_T_sha256"], "mean_w_c_numpy": float(wc_np.mean())}
check("w_c_digest_torch_equals_lock", OUT["w_c"]["digest_torch"] == geo["w_c_sha256"])
check("w_hat_digest_equals_lock", OUT["w_c"]["w_hat_digest_torch"] == geo["w_hat_sha256"])
check("d_T_digest_equals_lock", OUT["w_c"]["d_T_digest"] == geo["d_T_sha256"])
check("w_c_norm_about_0766", abs(norm_np - geo["w_c_norm"]) <= 1e-15 and abs(geo["w_c_norm"] - 0.766) < 0.001, [norm_np, geo["w_c_norm"]])
raw = mine.read_weight_from_raw(weights, d_T)
rw_lock = np.array(l011["read_weight"], dtype=np.float64)
wc_raw = raw["read_weight"] - math.fsum(raw["read_weight"].tolist()) / 512
OUT["w_c_from_raw_weights"] = {"max_abs_read_weight_raw_vs_011": float(np.max(np.abs(raw["read_weight"] - rw_lock))),
                               "max_abs_w_c_raw_vs_011_centred": float(np.max(np.abs(wc_raw - wc_np))), "tolerance_G1": 1e-9}
check("G1_from_raw_weights_within_1e-9", OUT["w_c_from_raw_weights"]["max_abs_w_c_raw_vs_011_centred"] <= 1e-9, OUT["w_c_from_raw_weights"])

# 12. The isotropic directions: 36 frames x 8, mine (torch-arithmetic and numpy-arithmetic variants), digests and checks.
frame_ids = [f["frame_id"] for f in conf["frames"]]
check("iso_frames_are_the_frozen_frames_in_order", list(geo["iso_sha256"]) == frame_ids and all(len(v) == 8 for v in geo["iso_sha256"].values()))
check("iso_tag_and_k", geo["iso_tag"] == "101|iso|{frame_id}|{j}" and geo["k_iso"] == 8)
w_hat_np = wc_np / norm_np
ones = np.full(512, 1.0 / math.sqrt(512))
digest_mismatch_torch, digest_mismatch_np = [], []
worst = {"norm": 0.0, "mean": 0.0, "dot_w": 0.0, "dot_ones": 0.0, "np_vs_torch": 0.0, "pairwise_abs_cos_max": 0.0}
for fid in frame_ids:
    vecs_np = []
    for j in range(1, 9):
        v_t = mine.iso_torch(what_t, fid, j)
        v_n = mine.iso_numpy(w_hat_np, fid, j)
        vecs_np.append(v_n)
        if mine.tensor_digest_torch(v_t) != geo["iso_sha256"][fid][j - 1]:
            digest_mismatch_torch.append(f"{fid}|{j}")
        if mine.tensor_digest_np(v_n) != geo["iso_sha256"][fid][j - 1]:
            digest_mismatch_np.append(f"{fid}|{j}")
        worst["norm"] = max(worst["norm"], abs(math.sqrt(math.fsum(x * x for x in v_n.tolist())) - 1.0))
        worst["mean"] = max(worst["mean"], abs(math.fsum(v_n.tolist()) / 512))
        worst["dot_w"] = max(worst["dot_w"], abs(math.fsum((v_n * w_hat_np).tolist())))
        worst["dot_ones"] = max(worst["dot_ones"], abs(math.fsum((v_n * ones).tolist())))
        worst["np_vs_torch"] = max(worst["np_vs_torch"], float(np.max(np.abs(v_n - v_t.numpy()))))
    gram = np.array(vecs_np) @ np.array(vecs_np).T
    worst["pairwise_abs_cos_max"] = max(worst["pairwise_abs_cos_max"], float(np.max(np.abs(gram - np.diag(np.diag(gram))))))
OUT["iso"] = {"digest_mismatches_torch_arith": digest_mismatch_torch, "digest_mismatches_numpy_arith": len(digest_mismatch_np), "worst": worst, "n": len(frame_ids) * 8}
check("iso_digests_all_288_equal_lock (my torch-arithmetic implementation)", not digest_mismatch_torch, digest_mismatch_torch[:5])
check("iso_unit_meanzero_orthogonal_1e-12", worst["norm"] <= 1e-12 and worst["mean"] <= 1e-12 and worst["dot_w"] <= 1e-12, worst)
OUT["g1_lock"] = geo["g1"]
check("g1_values_in_lock_within_tolerance", geo["g1"]["read_weight_max_abs"] <= 1e-9 and all(geo["g1"][k] <= 1e-12 for k in ("basis_norms", "basis_orthogonal", "v_norm", "v_mean", "v_dot_w")))
basis_np = {"ones_norm_minus_1": abs(float(np.linalg.norm(ones)) - 1.0), "w_hat_norm_minus_1": abs(float(np.linalg.norm(w_hat_np)) - 1.0),
            "ones_dot_w_hat": abs(float(ones @ w_hat_np))}
OUT["basis_mine"] = basis_np
check("u_off_rule_text", geo["u_off_rule"].startswith("per pair at run time, from the plain runs of the same frame: Δx = x_w − x_ref at RESID_PRE.L3, p_c; Δr = ⟨Δx, w_c⟩; "
                                                      "read = (Δr/⟨w_c, w_c⟩)·w_c; off_c = (Δx − read) − mean(Δx − read)·𝟙; û_off = off_c/‖off_c‖; s = |Δr|/‖w_c‖"))

# 13. The stage-1 binding: my layout from the frozen frames, d_model from the checkpoint, heads from config.json, nouns from 020's c_ref.
config_json = json.loads((SNAPSHOT / "config.json").read_text())
n_heads, d_model = int(config_json["num_attention_heads"]), int(emb.shape[1])
c_ref_lengths = {len(entry["c_ref"]) for entry in locked.values()}
layout = mine.my_layout(conf["frames"], d_model, n_heads, 80)
st = lock["stage1"]
check("stage1_paths", st["paths"] == {"states_data": f"{EXP}/stage1/states.f64", "states_index": f"{EXP}/stage1/states.json", "table": f"{EXP}/stage1/table.json"})
check("stage1_layout_equals_mine", st["states"]["layout"] == layout and len(layout) == 324, {"n": len(st["states"]["layout"]), "d_model": d_model, "n_heads": n_heads,
                                                                                            "c_ref_lengths_in_020": sorted(c_ref_lengths)})
check("stage1_meta", st["states"]["meta"] == {"experiment": "101", "kind": "the stage-1 reference states",
                                                "fields": ["x1_all", "x2_all", "x3_all", "x4_all", "x5_all", "h6", "rows4", "rows5", "c_ref"],
                                                "construction": "rd.locked_state of each S1-REF capture (rd.capture_frame_020), float64; rebuilt with rd.state_from_locked",
                                                "frames": [{"frame_id": f["frame_id"], "p_c": f["p_c"], "p_t": f["p_t"]} for f in conf["frames"]]})
# the layout's field shapes against 020's locked states (the same rd.locked_state), frame by frame
shape_mismatch = []
for fid, entry in locked.items():
    shapes = mine.locked_entry_shapes(entry)
    keys = entry["p_t"] + 1
    want = {"x1_all": [keys, 512], "x2_all": [keys, 512], "x3_all": [keys, 512], "x4_all": [keys, 512], "x5_all": [keys, 512], "h6": [512], "rows4": [8, keys],
            "rows5": [8, keys], "c_ref": [80]}
    if shapes != want or set(entry) != set(want) | {"p_c", "p_t"}:
        shape_mismatch.append(fid)
check("layout_formula_matches_020_locked_states (108 entries)", not shape_mismatch, shape_mismatch[:3])
tm = st["table"]
check("stage1_table_meta", tm == {"experiment": "101", "schema_version": 1, "kind": "the program's prediction table (computation 6)", "construction": tm["construction"],
                                  "mask": 30, "fractions_for_weak": False, "weak_shift": -1.0, "noun_keys": lock["noun_keys"],
                                  "arms": ["PL", "NEC-READ", "SUF-READ", "NEC-OFF", "SUF-OFF", "NEC-AMP", "SUF-NEG", "NEC-HALF", "SUF-HALF", "PC-SWAP-PL", "PC-SWAP-REF"],
                                  "cue_order": ["two", "three", "many"], "reference_id": 581})
check("stage1_freeze_text", st["freeze"].startswith("written once at stage 1 directly at these paths"))

# 14. Statistics, constants, tolerances, outcome, semantics.
check("tolerances_equal_plan_table", lock["tolerances"] == mine.EXPECTED_TOLERANCES)
c = lock["constants"]
check("constants", (c["weak_shift"], c["holm_alpha"], c["median_floor"], c["tau_cap"], c["tau_floor"], c["head_share_guard"], c["bootstrap"], c["bootstrap_seed"],
                    c["interval_ranks"]) == (-1.0, "1/40", 0.05, 0.2, 0.02, 0.05, 10000, 20261001, [math.ceil(0.025 * 10000), math.ceil(0.975 * 10000)]))
check("worst_values", {k: c["worst"][f"{k}_nonfinite"] for k in ("N", "S", "A_neg", "A_amp", "head_share", "iso_margin", "F_head_rem", "F_head_ins")}
      == {"N": "-inf", "S": "-inf", "A_neg": "-inf", "A_amp": "+inf", "head_share": "-inf", "iso_margin": "-inf", "F_head_rem": "+inf", "F_head_ins": "-inf"}
      and all(c["worst"][k] is None for k in ("N", "S", "A_neg", "A_amp", "head_share", "iso_margin", "F_head_rem", "F_head_ins")))
check("direction_free", c["direction_free"] == ["F_half_rem", "F_half_ins", "F_swap_pl", "F_swap_ref", "additivity", "P_error_rem", "P_error_ins"])
outcome = lock["outcome"]
check("outcome_labels_order", [row[1] for row in outcome["table"]] == [label for _, label in mine.DESIGN_OUTCOME_TABLE] == outcome["labels"])
check("outcome_conditions_match_design", [row[0] for row in outcome["table"]][1:] == [cond for cond, _ in mine.DESIGN_OUTCOME_TABLE][1:]
      and outcome["table"][0][0] == "any gate fails, or an incident is recorded")
sem = lock["semantics"]
design_scope = ("the read component of the residual entering layer 3 at the cue position, as every reader of that residual sees it, in coordinated frames, "
                "for the cue pair one/two from which the read was estimated")
check("scope_equals_design_7", sem["scope"] == design_scope)
OUT["fs_events"] = EVENTS
OUT["problems"] = problems
write_json("b_independent.json", OUT)
for name, entry in OUT["checks"].items():
    print(("PASS " if entry["ok"] else "FAIL ") + name + ("" if entry["ok"] or "detail" not in entry else f"  -> {json.dumps(entry['detail'], default=str)[:600]}"))
print("w_c:", json.dumps(OUT["w_c"], indent=1))
print("w_c from raw:", OUT["w_c_from_raw_weights"])
print("iso:", json.dumps(OUT["iso"], indent=1))
print("basis mine:", OUT["basis_mine"], "g1 lock:", OUT["g1_lock"])
print("params by HF names:", OUT["parameters_digest_by_hf_names"])
print("tau:", OUT["tau_recompute"], "holm:", OUT["holm"])
print("fs events:", json.dumps(EVENTS))
print("PROBLEMS:", problems)
