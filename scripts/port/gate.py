# WorkspaceBench capability gate (wsbench capable) run locally: smaller models answer each family's own
# bank question; graded by a local Qwen3.6-27B (the bench's GRADE prompt) and by string match.
# Mirrors src/wsbench/capable/run.py: 10 draws at T=0.7 + 1 greedy, item passes when greedy is right AND
# sampled rate >= threshold (0.8; 1.0 for chain/brew) with >= min_decided decided draws.
# Deviations from the bench: local HF generation instead of OpenRouter, no JSON-schema wrapper on answers,
# grader = Qwen3.6-27B (bench: Gemini 3.8 Flash) asked for JSON in-prompt.
# Few-shot format (key prefix "fs-", e.g. fs-q35_9b_base): raw text, ANSWER_SYSTEM header + N_SHOT solved items of the same
# family (excluded from scoring), "Question: ...\nAnswer:", stop at newline. Same prompt for base and instruct models.
# usage: python gate.py answer <model_key> | grade <shard> <nshards> <model_key,...> | table <model_key,...>
import functools, hashlib, json, os, random, re, sys, time, unicodedata
from collections import defaultdict
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import replace
from wsbench.capable.questions import BUILDERS, bank_items, threshold_for
from wsbench.capable.questions import build as bench_build
from wsbench.capable.run import ANSWER_SYSTEM, GRADE_SYSTEM, DEFAULT_DRAWS, DEFAULT_TEMPERATURE, DEFAULT_THRESHOLD, render_grade, min_decided

MODELS = {
    "q36_27b": "Qwen/Qwen3.6-27B",  # harness control: banks were selected on this model
    "q25_3b": "Qwen/Qwen2.5-3B-Instruct",
    "q35_4b": "Qwen/Qwen3.5-4B",
    "q35_9b": "Qwen/Qwen3.5-9B",
    "g3_12b": "google/gemma-3-12b-it",
    "q25_3b_base": "Qwen/Qwen2.5-3B",
    "q25_7b": "Qwen/Qwen2.5-7B-Instruct",
    "q25_7b_base": "Qwen/Qwen2.5-7B",
    "q35_4b_base": "Qwen/Qwen3.5-4B-Base",
    "q35_9b_base": "Qwen/Qwen3.5-9B-Base",
    "g4_12b": "google/gemma-4-12B-it",
    "g4_12b_base": "google/gemma-4-12B",
}
GRADER = "Qwen/Qwen3.6-27B"
FAMILIES = list(BUILDERS)  # the 21 families with an answerable gate question
DRAWS, TEMP, SEED = DEFAULT_DRAWS, DEFAULT_TEMPERATURE, 0
SAMPLE_KW = dict(top_p=1.0, top_k=0, repetition_penalty=1.0)  # pure temperature sampling, overrides model generation_config
MAX_NEW, GRADE_MAX_NEW = 128, 64
MAX_SEQS = int(os.environ.get("MAX_SEQS", 320))  # sequences per generate call; 27B needs ~96 (linear-attn state ~150MB/seq fp32)
EXACT = {"moral_rationale", "arithmetic_intermediates", "chain_intermediates", "brew_intermediates"}
N_SHOT, SHOT_SEED, FS_STOP = 3, 0, "\n"  # few-shot: exemplars per family, seed of their draw, answer ends at first newline
GRADE_STOP = ["true", "false"]  # grader verdict is the first JSON field; stopping there gives the same greedy prefix
GRADE_SUFFIX = '\n\nReply with a JSON object only: {"correct": true or false, "why": "<one short sentence>"}'
OUT = Path("/workspace/wsb_out/gate")
THINK = re.compile(r"<think>.*?</think>", re.S)
# Bench bug: capable/questions.py golds these families with `intermediates`, but the question asks for the
# final number/colour (bank field `answer`, which the banks' own gates used). Grade against `answer`.
FINAL_ANSWER = {"arithmetic_intermediates", "chain_intermediates", "brew_intermediates"}


def build(family):
    qs = bench_build(family)
    if family not in FINAL_ANSWER:
        return qs
    ans = {it["id"]: str(it["answer"]) for it in bank_items(family)}
    return [replace(q, golds=[ans[q.id]]) for q in qs]


@functools.cache
def shots(family):
    # exemplars get distinct answers where the family allows (else e.g. 3x "false" primes moral_rationale)
    qs, rng = build(family), random.Random(f"{SHOT_SEED}|{family}")
    need = min(N_SHOT, len({q.golds[0] for q in qs}))
    ex = rng.sample(qs, N_SHOT)
    while len({q.golds[0] for q in ex}) < need:
        ex = rng.sample(qs, N_SHOT)
    assert all("\n" not in q.golds[0] for q in ex), family
    return ex


def excluded(family):
    # exemplars plus their sibling hops (same id before ":"), which can share the scored answer
    base = {q.id.split(":")[0] for q in shots(family)}
    return {q.id for q in build(family) if q.id.split(":")[0] in base}


def fs_ids(tok, family, q):
    text = ANSWER_SYSTEM + "\n\n" + "".join(f"Question: {s.ask}\nAnswer: {s.golds[0]}\n\n" for s in shots(family)) + f"Question: {q.ask}\nAnswer:"
    return tok(text)["input_ids"]


def ans_files(mk):
    return [p for p in sorted((OUT / "answers").glob("*.jsonl")) if re.fullmatch(rf"{re.escape(mk)}(_p\d+)?\.jsonl", p.name)]


def load(name):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda:0").eval()
    return tok, model


def chat_ids(tok, system, user):
    out = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                  tokenize=True, add_generation_prompt=True, enable_thinking=False)
    return list(out["input_ids"] if hasattr(out, "keys") else out)


def generate(tok, model, prompts, max_new, draws, sample, stop=None):
    # prompts: [(key, ids)]; bucketed by exact length so no padding is ever needed. Returns {key: [(text, truncated)]}
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    pad = tok.pad_token_id if tok.pad_token_id is not None else min(eos)
    kw = dict(do_sample=True, temperature=TEMP, num_return_sequences=draws, **SAMPLE_KW) if sample else dict(do_sample=False, top_p=None, top_k=None, temperature=None)
    if stop:
        kw |= dict(stop_strings=stop, tokenizer=tok)
    buckets = defaultdict(list)
    for k, ids in prompts:
        buckets[len(ids)].append((k, ids))
    res, per = {}, max(1, MAX_SEQS // draws)
    for L, items in sorted(buckets.items()):
        for i in range(0, len(items), per):
            chunk = items[i:i + per]
            x = torch.tensor([ids for _, ids in chunk], device="cuda:0")
            with torch.no_grad():
                out = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=max_new, pad_token_id=pad, **kw)
            gen = out[:, L:].tolist()
            for j, (k, _) in enumerate(chunk):
                rows = gen[j * draws:(j + 1) * draws]
                if stop == [FS_STOP]:  # few-shot answer: text up to the first newline; truncated = never reached newline or eos
                    txt = [tok.decode(r, skip_special_tokens=True).lstrip(" ") for r in rows]
                    res[k] = [(t.split(FS_STOP)[0].strip(), FS_STOP not in t and not (eos & set(r))) for t, r in zip(txt, rows)]
                else:
                    res[k] = [(THINK.sub("", tok.decode(r, skip_special_tokens=True)).strip(), len(r) == max_new and not (eos & set(r))) for r in rows]
    return res


def norm(s):
    s = unicodedata.normalize("NFKC", s).casefold()
    s = "".join(c if not unicodedata.category(c).startswith(("P", "S")) else " " for c in s)
    return " ".join(s.split())


def string_correct(family, golds, ans):
    a = norm(ans)
    if not a:
        return False
    if family in EXACT:
        return a in {norm(g) for g in golds}
    return any(norm(g) and re.search(r"(?<!\w)" + re.escape(norm(g)) + r"(?!\w)", a) for g in golds)


def gkey(family, qid, ans):
    return hashlib.sha256(f"{family}|{qid}|{ans}".encode()).hexdigest()[:20]


def jl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def answer(mk, fams=None, part=""):
    torch.manual_seed(SEED)
    fs = mk.startswith("fs-")
    tok, model = load(MODELS[mk.removeprefix("fs-")])
    gc = model.generation_config
    print("model", mk, MODELS[mk.removeprefix("fs-")], type(model).__name__, "gen_config defaults", {k: getattr(gc, k, None) for k in ("temperature", "top_p", "top_k", "repetition_penalty", "eos_token_id")}, "override", SAMPLE_KW, "T", TEMP)
    out = OUT / "answers" / f"{mk}{part}.jsonl"; out.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["family"], r["id"]) for p in ans_files(mk) for r in jl(p)}
    print("max_seqs", MAX_SEQS, "families", fams or "all", "already done", len(done), "->", out, "format", "fewshot" if fs else "chat")
    for f in fams or FAMILIES:
        ex = excluded(f) if fs else set()
        qs = [q for q in build(f) if (f, q.id) not in done and q.id not in ex]
        if not qs:
            continue
        prompts = [(q.id, fs_ids(tok, f, q) if fs else chat_ids(tok, ANSWER_SYSTEM, q.ask)) for q in qs]
        if f == (fams or FAMILIES)[0]:
            print("first ids", prompts[0][1][:6], "rendered prompt:", repr(tok.decode(prompts[0][1])))
        if fs:
            print(f, "exemplars excluded", sorted(ex), "of", len(build(f)))
        t = time.time()
        stop = [FS_STOP] if fs else None
        g = generate(tok, model, prompts, MAX_NEW, 1, False, stop)
        s = generate(tok, model, prompts, MAX_NEW, DRAWS, True, stop)
        with out.open("a") as fh:
            for q in qs:
                fh.write(json.dumps({"family": f, "id": q.id, "greedy": g[q.id][0][0], "samples": [a for a, _ in s[q.id]],
                                     "n_trunc": sum(tr for _, tr in g[q.id] + s[q.id])}, ensure_ascii=False) + "\n")
        print(mk, f, "items", len(qs), "sec", time.time() - t, "trunc", sum(tr for q in qs for _, tr in g[q.id] + s[q.id]),
              "| greedy[0]", repr(g[qs[0].id][0][0][:80]), "golds", qs[0].golds[:2], flush=True)


def grade(shard, nshards, mks):
    qmap = {f: {q.id: q for q in build(f)} for f in FAMILIES}
    out = OUT / "grades" / f"shard{shard}.jsonl"; out.parent.mkdir(parents=True, exist_ok=True)
    have = {r["key"] for p in (OUT / "grades").glob("shard*.jsonl") for r in jl(p)}
    todo = {}
    for mk in mks:
        for r in (r for p in ans_files(mk) for r in jl(p)):
            for a in [r["greedy"]] + r["samples"]:
                k = gkey(r["family"], r["id"], a)
                if a and k not in have and int(k, 16) % nshards == shard:
                    todo[k] = (r["family"], r["id"], a)
    tok, model = load(GRADER)
    prompts = [(k, chat_ids(tok, GRADE_SYSTEM, render_grade(qmap[f][qid], a) + GRADE_SUFFIX)) for k, (f, qid, a) in todo.items()]
    print("grader", GRADER, "models", mks, "shard", shard, "of", nshards, "distinct answers to grade", len(prompts))
    if prompts:
        print("rendered grade prompt:", repr(tok.decode(prompts[0][1])))
    t = time.time()
    for i in range(0, len(prompts), 4000):  # flush in blocks so a crash keeps progress
        res = generate(tok, model, prompts[i:i + 4000], GRADE_MAX_NEW, 1, False, GRADE_STOP)
        with out.open("a") as fh:
            for k, _ in prompts[i:i + 4000]:
                raw = res[k][0][0]
                m = re.search(r'"correct"\s*:\s*(true|false)', raw)
                f, qid, a = todo[k]
                fh.write(json.dumps({"key": k, "family": f, "id": qid, "answer": a, "correct": None if m is None else m.group(1) == "true", "raw": raw}, ensure_ascii=False) + "\n")
        print("graded", min(i + 4000, len(prompts)), "of", len(prompts), "sec", time.time() - t, flush=True)


def gradecheck(n):
    # re-grade n already-graded answers with GRADE_STOP; verdicts must match the full-length grades
    qmap = {f: {q.id: q for q in build(f)} for f in FAMILIES}
    old = [r for p in sorted((OUT / "grades").glob("shard*.jsonl")) for r in jl(p) if r["correct"] is not None]
    old = random.Random(0).sample(old, n)
    tok, model = load(GRADER)
    prompts = [(r["key"], chat_ids(tok, GRADE_SYSTEM, render_grade(qmap[r["family"]][r["id"]], r["answer"]) + GRADE_SUFFIX)) for r in old]
    t = time.time()
    res = generate(tok, model, prompts, GRADE_MAX_NEW, 1, False, GRADE_STOP)
    new = {k: (lambda m: None if m is None else m.group(1) == "true")(re.search(r'"correct"\s*:\s*(true|false)', v[0][0])) for k, v in res.items()}
    agree = sum(new[r["key"]] == r["correct"] for r in old)
    print("gradecheck n", n, "agree", agree, "unparsed", sum(v is None for v in new.values()), "old true frac", sum(r["correct"] for r in old) / n, "sec", time.time() - t, "example", repr(res[old[0]["key"]][0][0]))


def item_pass(verdicts, greedy_ok, thr):
    decided = [v for v in verdicts if v is not None]
    rate = sum(decided) / len(decided) if decided else None
    if rate is None or len(decided) < min_decided(DRAWS, thr) or greedy_ok is None:
        return rate, None
    return rate, rate >= thr and greedy_ok is not False


def table(mks):
    qmap = {f: {q.id: q for q in build(f)} for f in FAMILIES}
    G = {r["key"]: r["correct"] for p in (OUT / "grades").glob("shard*.jsonl") for r in jl(p)}
    res, conf = {}, defaultdict(lambda: [0, 0, 0, 0])  # conf: [both, llm_only, str_only, neither] over distinct graded answers
    for mk in mks:
        rows = [r for p in ans_files(mk) for r in jl(p)]
        by = defaultdict(list)
        for r in rows:
            by[r["family"]].append(r)
        for f in FAMILIES:
            thr = threshold_for(f, DEFAULT_THRESHOLD)
            stat = defaultdict(list)
            for r in by[f]:
                q = qmap[f][r["id"]]
                llm = lambda a: (False if not a else G.get(gkey(f, r["id"], a)))
                st = lambda a: string_correct(f, q.golds, a)
                for name, fn in (("llm", llm), ("str", st)):
                    rate, ok = item_pass([fn(a) for a in r["samples"]], fn(r["greedy"]), thr)
                    stat[name + "_gate"].append(ok); stat[name + "_acc"].append(rate); stat[name + "_greedy"].append(fn(r["greedy"]))
                if isinstance(q.reference, float):
                    stat["ref"].append(q.reference); stat["ref_local"].append(stat["llm_acc"][-1])
                stat["trunc"].append(r["n_trunc"])
                for a in set([r["greedy"]] + r["samples"]):
                    v = llm(a)
                    if a and v is not None:
                        conf[(mk, f)][0 if v and st(a) else 1 if v else 2 if st(a) else 3] += 1
            mean = lambda xs: (lambda d: sum(d) / len(d) if d else None)([float(x) for x in xs if x is not None])
            res[(mk, f)] = {k: mean(v) for k, v in stat.items() if k not in ("trunc",)} | {
                "n": len(by[f]), "undecided_llm": sum(x is None for x in stat["llm_gate"]), "trunc": sum(stat["trunc"]),
                "ref_mad": mean([abs(a - b) for a, b in zip(stat["ref"], stat["ref_local"]) if b is not None]) if stat["ref"] else None}
    n_unparsed = sum(v is None for v in G.values())
    print("grades", len(G), "unparseable", n_unparsed)
    fmt = lambda v: "-" if v is None else str(round(v, 2))
    print("family | thr | " + " | ".join(f"{mk} gate llm/str (acc)" for mk in mks))
    for f in FAMILIES:
        print(f, "|", threshold_for(f, DEFAULT_THRESHOLD), "|", " | ".join(f"{fmt(res[(mk, f)]['llm_gate'])}/{fmt(res[(mk, f)]['str_gate'])} ({fmt(res[(mk, f)]['llm_acc'])})" for mk in mks))
    print("per (model,family): n, undecided_llm, trunc, ref mean, ref_local mean, ref_mad, conf [both, llm_only, str_only, neither]")
    for mk in mks:
        for f in FAMILIES:
            r = res[(mk, f)]
            print(mk, f, r["n"], r["undecided_llm"], r["trunc"], fmt(r.get("ref")), fmt(r.get("ref_local")), fmt(r["ref_mad"]), conf[(mk, f)])
    json.dump({f"{mk}|{f}": v for (mk, f), v in res.items()} | {"conf": {f"{mk}|{f}": v for (mk, f), v in conf.items()}}, open(OUT / f"table_{'_'.join(mks)}.json", "w"), indent=1)


if __name__ == "__main__":
    mode = sys.argv[1]
    print("mode", sys.argv[1:], "families", len(FAMILIES), "draws", DRAWS, "T", TEMP, "seed", SEED, "max_new", MAX_NEW, "grader", GRADER, "n_shot", N_SHOT, "shot_seed", SHOT_SEED, "grade_stop", GRADE_STOP)
    if mode == "answer":
        answer(sys.argv[2], sys.argv[3].split(",") if len(sys.argv) > 3 else None, sys.argv[4] if len(sys.argv) > 4 else "")
    elif mode == "grade":
        grade(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4].split(","))
    elif mode == "gradecheck":
        gradecheck(int(sys.argv[2]))
    else:
        table(sys.argv[2].split(","))
