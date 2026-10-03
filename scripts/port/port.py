# WorkspaceBench port: regenerate the model-specific golds for a new subject model (Qwen3.5-9B first).
# gen <mk>      subject model: moral sides (10 samples + greedy) and one explanation per sample (follow-up turn
#               conditioned on that sample's own answer); basic_readout implicit favourites; directed_modulation
#               compliance (free generation); poetry plain-render greedy continuation.
# extract <mk>  Qwen3.6-27B: cluster each moral item's explanations into reasons (text, supports, explanation ids),
#               then judge whether the subject's top committed reason is the bank's look_for_reasons[0].
# Sampling = gate.py (T=0.7, top_p=1, top_k=0, 10 draws + greedy, thinking off). All chat renders: no system prompt,
# the bench's own read render (tplcheck.py: Qwen3.5-9B renders every family byte-identically to Qwen3.6-27B).
import json, re, sys, time
from pathlib import Path
import torch
from gate import MODELS, GRADER, DRAWS, SEED, TEMP, SAMPLE_KW, MAX_SEQS, THINK, load, generate, norm, jl
from wsbench.capable.questions import bank_items
from wsbench.readplan import ANSWER_STEM

OUT = Path("/workspace/wsb_out/port")
EXPLAIN = "In one or two sentences, what is the main reason for your answer?"
SIDE = {"yes": "yes", "true": "yes", "no": "no", "false": "no"}
EXTRACT_SEQS = 24  # long prompts (~1.5k + 900 new): 96 OOMs the 27B on 80GB
MAX_NEW = dict(side=8, explain=120, implicit=8, dm=48, poetry=8, extract=900, match=8)
EXTRACT_SYSTEM = "You analyse short explanations a language model gave for its one-word answers to an ethical yes/no question."
EXTRACT_USER = """Question: {question}
Exact prompt the model answered: {stimulus}

The model's explanations, each written right after its own answer:
{listing}

Group these explanations into distinct reasons (considerations). Merge explanations that give the same consideration in different words; keep genuinely different considerations separate; an explanation may state more than one reason. For each reason give:
- "text": one sentence stating the consideration itself, e.g. "Paying ransom creates incentives that lead to more future kidnappings" (not "the model says...")
- "supports": "yes" if the reason argues for answering yes/true, "no" if it argues for no/false
- "explanations": the numbers of the explanations that state it

Reply with JSON only: {{"reasons": [{{"text": "...", "supports": "yes", "explanations": [1, 4]}}]}}"""
MATCH_SYSTEM = "You compare two statements of a reason behind an answer to an ethical question."
MATCH_USER = """Question: {question}
Reason A: {a}
Reason B: {b}

Do A and B state the same consideration (the same substance, not merely the same topic)? Answer with yes or no only."""


def chat(tok, msgs, prefill=""):
    out = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    ids = list(out["input_ids"] if hasattr(out, "keys") else out)
    return ids + (tok(prefill, add_special_tokens=False)["input_ids"] if prefill else [])


def gen_padded(tok, model, prompts, max_new, sample, seqs=MAX_SEQS, temp=TEMP):
    # left-padded batches sorted by length (bucketing by exact length is ~1 prompt per call for varied lengths)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    pad = tok.pad_token_id if tok.pad_token_id is not None else min(eos)
    kw = dict(do_sample=True, temperature=temp, **SAMPLE_KW) if sample else dict(do_sample=False, top_p=None, top_k=None, temperature=None)
    items, res = sorted(prompts, key=lambda p: len(p[1])), {}
    for i in range(0, len(items), seqs):
        chunk = items[i:i + seqs]
        L = max(len(ids) for _, ids in chunk)
        x = torch.tensor([[pad] * (L - len(ids)) + ids for _, ids in chunk], device="cuda:0")
        m = torch.tensor([[0] * (L - len(ids)) + [1] * len(ids) for _, ids in chunk], device="cuda:0")
        with torch.no_grad():
            out = model.generate(x, attention_mask=m, max_new_tokens=max_new, pad_token_id=pad, **kw)
        for (k, _), r in zip(chunk, out[:, L:].tolist()):
            r = r[:next((j + 1 for j, t in enumerate(r) if t in eos), len(r))]
            res[k] = (THINK.sub("", tok.decode(r, skip_special_tokens=True)).strip(), len(r) == max_new and not (eos & set(r)))
    return res


def first_word(s):
    w = norm(s).split()
    return w[0] if w else ""


def side_of(s):
    return SIDE.get(first_word(s), "other")


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", path, len(rows))


def sample_set(tok, model, prompts, max_new):
    # greedy + DRAWS samples per prompt: {key: (greedy_text, [sample_texts])}
    g = generate(tok, model, prompts, max_new, 1, False)
    s = generate(tok, model, prompts, max_new, DRAWS, True)
    return {k: (g[k][0][0], [a for a, _ in s[k]]) for k, _ in prompts}


def gen(mk):
    torch.manual_seed(SEED)
    tok, model = load(MODELS[mk])
    d = OUT / mk
    # moral: side, then one explanation per sampled answer
    moral = bank_items("moral_rationale")
    t = time.time()
    res = sample_set(tok, model, [(it["id"], chat(tok, [{"role": "user", "content": it["stimulus"]}])) for it in moral], MAX_NEW["side"])
    print("moral render[0]", repr(tok.decode(chat(tok, [{"role": "user", "content": moral[0]["stimulus"]}]))), "sec", time.time() - t)
    ex_prompts = [((it["id"], j), chat(tok, [{"role": "user", "content": it["stimulus"]}, {"role": "assistant", "content": a}, {"role": "user", "content": EXPLAIN}]))
                  for it in moral for j, a in enumerate(res[it["id"]][1])]
    print("explain render[0]", repr(tok.decode(ex_prompts[0][1])))
    ex = gen_padded(tok, model, ex_prompts, MAX_NEW["explain"], True)
    write(d / "moral.jsonl", [{"id": it["id"], "greedy": res[it["id"]][0], "samples": res[it["id"]][1],
                               "explanations": [ex[(it["id"], j)][0] for j in range(DRAWS)]} for it in moral])
    # basic_readout implicit: the bench's chat_prefill render with ANSWER_STEM
    imp = [it for it in bank_items("basic_readout") if it.get("subfamily") == "implicit"]
    res = sample_set(tok, model, [(it["name"], chat(tok, [{"role": "user", "content": it["prompt"]}], ANSWER_STEM)) for it in imp], MAX_NEW["implicit"])
    print("implicit render[0]", repr(tok.decode(chat(tok, [{"role": "user", "content": imp[0]["prompt"]}], ANSWER_STEM))))
    write(d / "implicit.jsonl", [{"id": k, "greedy": g, "samples": s} for k, (g, s) in res.items()])
    # directed_modulation: free generation from the user turn (the read render teacher-forces the carrier)
    dm = bank_items("directed_modulation")
    res = sample_set(tok, model, [(it["name"], chat(tok, [{"role": "user", "content": it["prompt"]}])) for it in dm], MAX_NEW["dm"])
    write(d / "dm.jsonl", [{"id": k, "greedy": g, "samples": s} for k, (g, s) in res.items()])
    # poetry: greedy continuation of the plain render the lens reads
    po = bank_items("poetry")
    g = generate(tok, model, [(it["name"], tok(it["prompt"], add_special_tokens=False)["input_ids"]) for it in po], MAX_NEW["poetry"], 1, False)
    write(d / "poetry_plain.jsonl", [{"id": k, "greedy": v[0][0]} for k, v in g.items()])


def parse_json(raw):
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def extract(mk):
    bank = {it["id"]: it for it in bank_items("moral_rationale")}
    rows = jl(OUT / mk / "moral.jsonl")
    tok, model = load(GRADER)
    prompts = []
    for r in rows:
        it = bank[r["id"]]
        listing = "\n".join(f"{j + 1}. [answered: {a.strip()[:20]}] {e.strip()}" for j, (a, e) in enumerate(zip(r["samples"], r["explanations"])) if side_of(a) != "other")
        prompts.append((r["id"], chat(tok, [{"role": "system", "content": EXTRACT_SYSTEM}, {"role": "user", "content": EXTRACT_USER.format(question=it["question"], stimulus=it["stimulus"], listing=listing)}])))
    print("extract prompt[0]", repr(tok.decode(prompts[0][1])))
    t = time.time()
    res = gen_padded(tok, model, prompts, MAX_NEW["extract"], False, EXTRACT_SEQS)
    out = []
    for r in rows:
        raw = res[r["id"]][0]
        js = parse_json(raw)
        out.append({"id": r["id"], "reasons": js.get("reasons") if js else None, "raw": raw, "truncated": res[r["id"]][1]})
    print("extract sec", time.time() - t, "unparsed", sum(o["reasons"] is None for o in out), "truncated", sum(o["truncated"] for o in out), "example", json.dumps(out[0]["reasons"])[:400])
    write(OUT / mk / "moral_reasons.jsonl", out)
    # top committed-side reason vs the bank's look_for_reasons[0]; pairs only where a reason exists on the bank's side
    pairs = []
    for r, o in zip(rows, out):
        it = bank[r["id"]]
        cand = [x for x in (o["reasons"] or []) if x.get("supports") == it["commit_direction"]]
        if cand:
            top = max(cand, key=lambda x: len(x.get("explanations") or []))
            pairs.append((r["id"], top["text"], it["look_for_reasons"][0]))
    mp = [(k, chat(tok, [{"role": "system", "content": MATCH_SYSTEM}, {"role": "user", "content": MATCH_USER.format(question=bank[k]["question"], a=a, b=b)}])) for k, a, b in pairs]
    res = gen_padded(tok, model, mp, MAX_NEW["match"], False)
    chk = generate(tok, model, mp[:24], MAX_NEW["match"], 1, False)
    print("padded vs bucketed greedy agree", sum(chk[k][0][0] == res[k][0] for k, _ in mp[:24]), "of", len(mp[:24]))
    write(OUT / mk / "moral_match.jsonl", [{"id": k, "ours": a, "bank": b, "same": first_word(res[k][0]) == "yes", "raw": res[k][0]} for k, a, b in pairs])


FLOOR_DRAWS, FLOOR_TEMP = 5, 1.0  # = wsbench baselines/lucky_guessing.py (blind variant, Gemini there, Qwen3.6-27B here)


def floor(banks):
    # blind lucky-guessing floor on moral_rationale banks: the guesser sees only the option lists (no question, no readout)
    from wsbench.evals.moral_rationale.judge import build_mcs, build_pools
    from wsbench.baselines.lucky_guessing import SYSTEM, Item, render_lists
    tok, model = load(GRADER)
    for spec in banks.split(","):
        tag, path = spec.split("=")
        bank = json.load(open(path))
        pools = build_pools(bank)
        items = []
        for it in bank:
            mcs = build_mcs(it, *pools)
            if mcs:
                sides = sorted(mcs)
                items.append(Item(it["id"], [mcs[x].shown[:-1] for x in sides], [mcs[x].gold_pos for x in sides]))
        ask = lambda it: render_lists(it) + "\n\nReply with JSON only: {" +", ".join(f'"choice{i + 1}": <option number>' for i in range(len(it.lists))) + "}"
        prompts = [((it.id, d), chat(tok, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ask(it)}])) for it in items for d in range(FLOOR_DRAWS)]
        res = gen_padded(tok, model, prompts, 40, True, temp=FLOOR_TEMP)
        acc = []
        for it in items:
            for d in range(FLOOR_DRAWS):
                js = parse_json(res[(it.id, d)][0]) or {}
                acc.append(all(js.get(f"choice{i + 1}") == g for i, g in enumerate(it.golds)))
        n_opt = sorted({len(l) for it in items for l in it.lists})
        print("floor", tag, "items", len(items), "draws", FLOOR_DRAWS, "T", FLOOR_TEMP, "options per list", n_opt, "blind mean acc", sum(acc) / len(acc), "example", repr(res[(items[0].id, 0)][0]))


if __name__ == "__main__":
    print("mode", sys.argv[1:], "draws", DRAWS, "seed", SEED, "max_new", MAX_NEW, "explain", EXPLAIN)
    {"gen": gen, "extract": extract, "floor": floor}[sys.argv[1]](sys.argv[2])
