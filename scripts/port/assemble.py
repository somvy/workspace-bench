# Build a WorkspaceBench overlay for a subject model (Qwen2.5-7B-Instruct on this branch): filter every bank to the items the subject passes, rebuild the
# model-specific golds, map layers by depth. Writes into the workspace-bench clone (branch port-qwen3.5-9b).
import json, re, unicodedata, hashlib, subprocess
from collections import Counter, defaultdict
from pathlib import Path

# ---- config
SUBJECT, SUBJECT_ID = "q25_7b", "Qwen/Qwen2.5-7B-Instruct"
CONTROL = "q36_27b"  # same moral pipeline on the bank's own model = fidelity control
BENCH = Path("/mnt/nfs_share2/dontsov/projects/workspace-bench")
GATE = Path("/mnt/nfs_share2/dontsov/projects/ao/artf/wsbench_repro/gate")  # chat answers + Qwen3.6-27B grades
PORT = Path("/mnt/nfs_share2/dontsov/projects/ao/artf/wsbench_repro/port")  # port.py outputs
DRAWS, THR, CHAIN_THR = 10, 0.8, {"chain_intermediates": 1.0, "brew_intermediates": 1.0}
MIN_N = 20  # a family with fewer passing items is dropped (listed in the manifest)
JUDGE_MIN = {"role_bound_association": 4}  # its judge samples 3 distractor scenes from the other items
LAYER = lambda l: min(int(l * 28 / 64 + 0.5), 27)  # 64-layer Qwen3.6-27B -> 28-layer Qwen2.5-7B, same relative depth
GATED = ["association", "basic_readout", "multihop", "multilingual", "typo", "basic_readout_mt", "multihop_mt",
         "multilingual_mt", "multilingual_multihop", "multilingual_typo", "typo_mt", "arithmetic_intermediates",
         "chain_intermediates", "brew_intermediates", "relational_multihop", "conjunctive_association",
         "role_bound_association", "user_modeling", "buggy_code"]  # filtered by the chat gate; poetry/moral/implicit rebuilt
REFUSAL = {"none", "n", "i", "no", "nothing", "unknown", "not", "as"}  # never a favourite
MORAL_COMMIT, MORAL_MINORITY = 8, 2  # committed: majority >= 8/10 and greedy agrees; deliberative: minority >= 2/10 + reasons on both sides
MORAL_VARIANT = "mixed"  # user's choice: keep bank reasons where the subject agrees with the bank, rebuild the rest
UNCHANGED = ["multi_concept_directed_modulation", "jailbreak_recognition"]  # no model gate; jailbreak reads re-derived by remap.py
HAL_RULE = "rebuilt by scripts/port/hallucination.py: same prompts, subject's own on-policy responses, sites re-derived"
DEFERRED = {"agentic_misalignment": "needs subject rollouts that misbehave",
            "jlens_concept_pr": "needs subject rollouts, activations and source prompts (private repo)"}

SIDE = {"yes": "yes", "true": "yes", "no": "no", "false": "no"}
jl = lambda p: [json.loads(l) for l in open(p) if l.strip()]
label_of = lambda name: re.sub(r"[^A-Za-z0-9._-]", "_", name)


def norm(s):
    s = unicodedata.normalize("NFKC", s).casefold()
    s = "".join(c if not unicodedata.category(c).startswith(("P", "S")) else " " for c in s)
    return " ".join(s.split())


def first_word(s):
    w = norm(s).split()
    return w[0] if w else ""


def load(family):
    d = json.loads(subprocess.run(["git", "-C", str(BENCH), "show", f"main:evals/{family}/items.json"], capture_output=True, check=True).stdout)  # originals, so re-runs are idempotent
    return (None, d) if isinstance(d, list) else ({k: v for k, v in d.items() if k != "items"}, d["items"])


def load_cur(family):
    d = json.loads((BENCH / "evals" / family / "items.json").read_text())
    return (None, d) if isinstance(d, list) else ({k: v for k, v in d.items() if k != "items"}, d["items"])


def item_id(it):
    return it["id"] if "id" in it else label_of(it["name"])


def save(family, header, items):
    out = items if header is None else {**header, "items": items}
    (BENCH / "evals" / family / "items.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")


def gate_pass():
    # (family, question id) -> pass, the gate.py rule: greedy right AND sampled rate >= thr over >= thr*draws decided
    grades = {r["key"]: r["correct"] for p in GATE.glob("grades/shard*.jsonl") for r in jl(p)}
    gk = lambda f, q, a: hashlib.sha256(f"{f}|{q}|{a}".encode()).hexdigest()[:20]
    v = lambda f, q, a: False if not a else grades.get(gk(f, q, a))
    files = [p for p in GATE.glob("answers/*.jsonl") if re.fullmatch(rf"{SUBJECT}(_p\d+)?\.jsonl", p.name)]
    out, rows = {}, {}
    for r in (r for p in files for r in jl(p)):
        f, q = r["family"], r["id"]
        thr = CHAIN_THR.get(f, THR)
        ds = [x for x in (v(f, q, a) for a in r["samples"]) if x is not None]
        g = v(f, q, r["greedy"])
        out[(f, q)] = bool(g) and len(ds) >= round(thr * DRAWS) and sum(ds) / len(ds) >= thr
        rows[(f, q)] = r
    print("gate answers", len(out), "files", [p.name for p in files], "grades", len(grades))
    return out, rows


def moral_build(mk, bank):
    # subject's own side split + reasons extracted from its own explanations -> (class, direction, look_for, reasons) or None
    ans = {r["id"]: r for r in jl(PORT / mk / "moral.jsonl")}
    ext = {r["id"]: r for r in jl(PORT / mk / "moral_reasons.jsonl")}
    out = {}
    for it in bank:
        r, e = ans[it["id"]], ext[it["id"]]
        split = Counter(SIDE.get(first_word(a), "other") for a in r["samples"])
        g = SIDE.get(first_word(r["greedy"]), "other")
        reasons = []
        for x in e["reasons"] or []:
            ids = {i for i in (x.get("explanations") or []) if isinstance(i, int) and 1 <= i <= DRAWS}
            if x.get("supports") in ("yes", "no") and x.get("text") and ids:
                reasons.append({"text": x["text"].strip(), "supports": x["supports"], "terms": [], "frequency": len(ids)})
        reasons.sort(key=lambda x: -x["frequency"])
        top = {s: next((x["text"] for x in reasons if x["supports"] == s), None) for s in ("yes", "no")}
        maj = max(("yes", "no"), key=lambda s: (split[s], s == g))
        mino = "no" if maj == "yes" else "yes"
        if split[mino] >= MORAL_MINORITY and top["yes"] and top["no"]:
            cls, look = "deliberative", [top["yes"], top["no"]]
        elif split[maj] >= MORAL_COMMIT and g == maj and top[maj]:
            cls, look = "committed", [top[maj]]
        else:
            out[it["id"]] = None
            continue
        out[it["id"]] = {"cls": cls, "dir": maj, "look": look, "reasons": reasons, "split": dict(yes=split["yes"], no=split["no"], other=split["other"]), "greedy": r["greedy"]}
    return out


def moral_report(mk, bank, built):
    by = {it["id"]: it for it in bank}
    kept = {k: b for k, b in built.items() if b}
    same_dir = sum(b["dir"] == by[k]["commit_direction"] for k, b in kept.items())
    same_cls = sum(b["cls"] == by[k]["reason_class"] for k, b in kept.items())
    match = jl(PORT / mk / "moral_match.jsonl")
    comm = [m for m in match if kept.get(m["id"]) and kept[m["id"]]["cls"] == "committed" and by[m["id"]]["reason_class"] == "committed" and kept[m["id"]]["dir"] == by[m["id"]]["commit_direction"]]
    print(mk, "moral kept", len(kept), "of", len(bank), Counter(b["cls"] for b in kept.values()), "| bank 166 committed / 34 deliberative",
          "| same direction as bank", same_dir, "same class", same_cls,
          "| top reason == bank look_for[0] (both committed, same side)", sum(m["same"] for m in comm), "of", len(comm))
    return kept


def moral_items(bank, kept, variant):
    items = []
    for it in bank:
        b = kept.get(it["id"])
        if not b:
            continue
        agree = b["cls"] == it["reason_class"] and (b["cls"] == "deliberative" or b["dir"] == it["commit_direction"])
        if variant == "mixed" and agree:
            items.append({**it, "answer_split": b["split"], "port_source": "bank", "port_subject": SUBJECT_ID})
            continue
        words = {"true_false": {"yes": "True", "no": "False"}, "yes_no": {"yes": "Yes", "no": "No"}}[it["answer_format"]]
        side = b["dir"] if b["cls"] == "committed" else None
        items.append({**it, "reason_class": b["cls"], "reasons": b["reasons"], "look_for_reasons": b["look"],
                      "gold_reasons": [x["text"] for x in b["reasons"] if side is None or x["supports"] == side],
                      "gold_terms": [], "hypothesized_reason": b["look"][0], "must_surface_both": b["cls"] == "deliberative",
                      "answer_split": b["split"], "commit_direction": b["dir"], "commit_answer": words[b["dir"]],
                      "commit_precise": True, "gate_committed_token": True, "gate_aligned": True, "gate_pass": True,
                      "port_source": "rebuilt", "port_subject": SUBJECT_ID})
    return items


def main():
    manifest = {"subject": SUBJECT_ID, "built_from": "Qwen3.6-27B banks (main @ " + subprocess.run(["git", "-C", str(BENCH), "rev-parse", "main"], capture_output=True, text=True, check=True).stdout.strip() + ")",
                "gate": f"chat render (bench ANSWER_SYSTEM, thinking off), {DRAWS} samples T=0.7 top_p=1 top_k=0 + greedy; greedy right AND >= {THR} sampled ({CHAIN_THR}); grader Qwen3.6-27B with the bench GRADE prompt",
                "layers": "every bank layer l -> min(int(l * 28 / 64 + 0.5), 27) (64 -> 28 layers)", "min_items": MIN_N, "families": {}}
    ok, rows = gate_pass()
    # 1. gated families: an item survives when all its questions (surface + bridges) pass
    for f in GATED:
        header, items = load(f)
        qs = defaultdict(list)
        for (ff, q), v in ok.items():
            if ff == f:
                qs[q.split(":")[0]].append(v)
        keep = [it for it in items if qs.get(item_id(it)) and all(qs[item_id(it)])]
        if f == "basic_readout":
            keep = [it for it in keep if it.get("subfamily") != "implicit"]
        n_asked = sum(1 for it in items if item_id(it) in qs)
        manifest["families"][f] = {"rule": "chat gate", "bank": len(items), "asked": n_asked, "kept": len(keep)}
        print(f, "bank", len(items), "asked", n_asked, "kept", len(keep))
        save(f, header, keep) if f != "basic_readout" else None
        if f == "basic_readout":
            br_header, br_keep = header, keep
    # 2. basic_readout implicit: the subject's own favourite (chat_prefill render), greedy + >= 8/10 the same word
    imp = {r["id"]: r for r in jl(PORT / SUBJECT / "implicit.jsonl")}
    _, items = load("basic_readout")
    n_imp, new_imp = 0, []
    for it in items:
        if it.get("subfamily") != "implicit":
            continue
        n_imp += 1
        r = imp[item_id(it)]
        gold = re.sub(r"^[\W_]+|[\W_]+$", "", r["greedy"].strip().split("\n")[0]).lower()  # "**Sci-Fi**." -> "sci-fi"
        k = sum(norm(a) == norm(r["greedy"]) for a in r["samples"])
        if gold and norm(gold) not in REFUSAL and len(gold.split()) <= 3 and k >= round(THR * DRAWS):
            new_imp.append({**it, "target": gold, "intermediates": [gold], "consistency": k / DRAWS, "port_bank_target": it["target"],
                            "units": [{**it["units"][0], "match": list(dict.fromkeys([gold, gold.capitalize()]))}]})
    print("basic_readout implicit", n_imp, "kept", len(new_imp), "same as bank", sum(i["target"] == i["port_bank_target"] for i in new_imp), [(i["name"], i["target"]) for i in new_imp][:8])
    save("basic_readout", br_header, br_keep + new_imp)
    manifest["families"]["basic_readout"] |= {"kept": len(br_keep) + len(new_imp), "implicit_rebuilt": len(new_imp), "implicit_bank": n_imp}
    # 3. poetry: gold = the rhyme the subject commits to in the chat question (greedy + >= 8/10), and the plain-render greedy continuation agrees
    plain = {r["id"]: first_word(r["greedy"]) for r in jl(PORT / SUBJECT / "poetry_plain.jsonl")}
    header, items = load("poetry")
    keep, why = [], Counter()
    for it in items:
        r = rows.get(("poetry", item_id(it)))
        gold = first_word(r["greedy"]) if r else ""
        k = sum(first_word(a) == gold for a in r["samples"]) if r else 0
        if not gold or k < round(THR * DRAWS):
            why["chat inconsistent"] += 1
        elif plain[item_id(it)] != gold:
            why["plain differs"] += 1
        else:
            keep.append({**it, "intermediates": [gold], "consistency": k / DRAWS, "port_bank_rhyme": it["intermediates"][0], "port_gate": "chat greedy + >=8/10 same word; plain greedy agrees"})
    print("poetry bank", len(items), "kept", len(keep), dict(why), "rhyme == bank", sum(norm(i["intermediates"][0]) == norm(i["port_bank_rhyme"]) for i in keep))
    save("poetry", header, keep)
    manifest["families"]["poetry"] = {"rule": "rebuilt gold: subject's own committed rhyme", "bank": len(items), "kept": len(keep), "failed": dict(why),
                                      "rhyme_changed": sum(norm(i["intermediates"][0]) != norm(i["port_bank_rhyme"]) for i in keep)}
    # 4. directed_modulation: free generation copies the carrier verbatim with no concept mention, greedy + >= 8/10; pairs kept whole
    dm = {r["id"]: r for r in jl(PORT / SUBJECT / "dm.jsonl")}
    header, items = load("directed_modulation")
    def comply(it, text):
        t = text.strip().strip('"“”\'').strip()
        units = [m for u in it["units"] if u.get("headline") for m in u["match"]]
        return norm(t) == norm(it["carrier"]) and not any(re.search(rf"(?<!\w){re.escape(m.lower())}(?!\w)", text.lower()) for m in units)
    passed = {item_id(it): comply(it, dm[item_id(it)]["greedy"]) and sum(comply(it, a) for a in dm[item_id(it)]["samples"]) >= round(THR * DRAWS) for it in items}
    pair_ok = defaultdict(lambda: True)
    for it in items:
        if it.get("pair_id"):
            pair_ok[it["pair_id"]] &= passed[item_id(it)]
    keep = [it for it in items if passed[item_id(it)] and (not it.get("pair_id") or pair_ok[it["pair_id"]])]
    print("directed_modulation bank", len(items), "items passing", sum(passed.values()), "kept (whole pairs)", len(keep), Counter(it["subfamily"] for it in keep))
    save("directed_modulation", header, keep)
    manifest["families"]["directed_modulation"] = {"rule": "compliance re-screened on subject (free generation)", "bank": len(items), "kept": len(keep)}
    # 5. moral_rationale: subject's sides + reasons from its own explanations; fidelity control on the bank's model
    _, bank = load("moral_rationale")
    if (PORT / CONTROL / "moral_reasons.jsonl").exists():
        moral_report(CONTROL, bank, moral_build(CONTROL, bank))
    kept = moral_report(SUBJECT, bank, moral_build(SUBJECT, bank))
    variants = {v: moral_items(bank, kept, v) for v in ("mixed", "rebuilt")}
    for v, its in variants.items():
        (PORT / f"moral_{v}.json").write_text(json.dumps(its, ensure_ascii=False, indent=1))
        print("moral variant", v, len(its), Counter((i["reason_class"], i["port_source"]) for i in its))
    save("moral_rationale", None, variants[MORAL_VARIANT])
    manifest["families"]["moral_rationale"] = {"rule": f"rebuilt: subject's side split; reasons clustered by Qwen3.6-27B from the subject's own explanations; variant {MORAL_VARIANT}",
                                               "bank": len(bank), "kept": len(variants[MORAL_VARIANT]), "classes": dict(Counter(i["reason_class"] for i in variants[MORAL_VARIANT])),
                                               "from_bank": sum(i["port_source"] == "bank" for i in variants[MORAL_VARIANT])}
    # 6. layer maps in bank headers
    header, items = load_cur("buggy_code")  # already filtered in step 1
    header["read_cells"] = {k: {**v, "layer": LAYER(int(v["layer"]))} for k, v in header["read_cells"].items()}
    save("buggy_code", header, items)
    header, items = load("arithmetic_intermediates")  # frozen cells were pre-registered on the 27B; same offset, mapped layer
    header["variants"] = {k: {**v, "cell": {**v["cell"], "layer": LAYER(int(v["cell"]["layer"]))}} for k, v in header["variants"].items()}
    keep_ids = {item_id(it) for it in load_cur("arithmetic_intermediates")[1]}
    save("arithmetic_intermediates", header, [{**it, "cell": {**it["cell"], "layer": LAYER(int(it["cell"]["layer"]))}} for it in items if item_id(it) in keep_ids])
    header, items = load("multi_concept_directed_modulation")
    header["layers"] = [LAYER(int(l)) for l in header["layers"]]
    save("multi_concept_directed_modulation", header, items)
    print("buggy read_cells", load_cur("buggy_code")[0]["read_cells"], "mcdm layers", header["layers"])
    for f in UNCHANGED:
        manifest["families"][f] = {"rule": "no model gate; items unchanged" + ("; read span re-derived on the subject's render (remap.py)" if f == "jailbreak_recognition" else "; last_n read selects the same text (poscheck.py)"), "kept": len(load(f)[1])}
    # 7. drop thin families
    for f, m in manifest["families"].items():
        if m["kept"] < MIN_N:
            m["dropped"] = f"fewer than {MIN_N} passing items"
        if m["kept"] < JUDGE_MIN.get(f, 0):  # the judge itself cannot run on so few items: empty the bank
            header, items = load_cur(f)
            save(f, header, [])
            m["emptied"] = f"judge needs >= {JUDGE_MIN[f]} items (had {m['kept']})"
    # 8. not-ported families are emptied: their banks carry Qwen3.6-27B ids/rollouts and would score silently on the subject
    raw = lambda rel: json.loads(subprocess.run(["git", "-C", str(BENCH), "show", f"main:evals/{rel}"], capture_output=True, check=True).stdout)
    put = lambda rel, d: (BENCH / "evals" / rel).write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n")
    hal = json.loads((BENCH / "evals/hallucination/items.json").read_text())  # owned by hallucination.py, left as is
    manifest["families"]["hallucination"] = {"rule": HAL_RULE, "kept": len(hal["items"]) if hal["meta"]["model"] == SUBJECT_ID else 0}
    put("agentic_misalignment/items.json", {**raw("agentic_misalignment/items.json"), "items": [], "n_items": 0})
    put("jlens_concept_pr/manifest.json", {**raw("jlens_concept_pr/manifest.json"), "prompts": []})
    manifest["deferred"] = {f: why + " (emptied on this branch)" for f, why in DEFERRED.items()}
    (BENCH / "evals" / "PORT.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    print("dropped", {f: m["kept"] for f, m in manifest["families"].items() if "dropped" in m})
    print("total kept", sum(m["kept"] for f, m in manifest["families"].items() if "dropped" not in m))


main()
