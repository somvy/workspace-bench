# Rebuild the hallucination bank for the subject: same 149 prompts, the subject's own on-policy responses, read sites
# re-derived on its tokens. The source repo's site code (hallucination_bench.sites) is not public; response_sites /
# select_sites below are a reimplementation from the bank's meta, and `check` proves it on the original bank.
# usage (vLLM venv): python hallucination.py check   -> reproduce Qwen3.6-27B's 1,123 sites from main's capture rows
#                    python hallucination.py build   -> generate with the subject, write evals/hallucination/{items,capture_rows}.json
import json, subprocess, sys, unicodedata
from pathlib import Path

# ---- config
SUBJECT_ID = "Qwen/Qwen3.5-9B"
ORIG_ID = "Qwen/Qwen3.6-27B"  # model of the original bank (main)
BENCH = Path(__file__).resolve().parents[2]
OUT = BENCH / "evals/hallucination"
SEED, TEMP, TOP_P, MAX_NEW = 7, 1.0, 1.0, 512  # original rollouts: T=1.0, top-p 1.0, top-k 0, 512 new tokens, seed 7
K = 8  # read sites per item after thinning
SITE_KINDS = ("clause", "sentence", "markup", "newline", "quote")


def is_site(s):  # a token made only of punctuation and/or whitespace, not pure spaces/tabs
    if not s.strip():
        return "\n" in s
    return all(c.isspace() or unicodedata.category(c).startswith("P") for c in s)


def kind(s):
    if "\n" in s:
        return "newline"
    if any(c in s for c in ".!?"):
        return "sentence"
    if any(c in s for c in ",;:-—–"):
        return "clause"
    if any(c in s for c in "\"'“”‘’"):
        return "quote"
    return "markup"


def response_sites(tok, gen):
    out = []
    for i, t in enumerate(gen):
        s = tok.decode([t])
        if is_site(s):
            out.append({"i": i, "token": s, "kind": kind(s), "char": len(tok.decode(gen[:i + 1]))})
    return out


def select_sites(c, k=K):  # evenly spread over the candidates, first one always kept
    return c if len(c) <= k else [c[int(j * len(c) / k)] for j in range(k)]


def sites_for(tok, ids, pl):
    return [{"pos": pl + s["i"], "token": s["token"], "kind": s["kind"], "char": s["char"]} for s in select_sites(response_sites(tok, ids[pl:]))]


def main_bank():
    raw = lambda rel: json.loads(subprocess.run(["git", "-C", str(BENCH), "show", f"main:evals/hallucination/{rel}"], capture_output=True, check=True).stdout)
    return raw("items.json"), {r["id"]: r for r in raw("capture_rows.json")}


def render(tok, prompt):  # chat template with an empty think block, as the original rollouts
    return tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, enable_thinking=False, tokenize=False)


def main():
    mode = sys.argv[1]
    from transformers import AutoTokenizer

    bank, cap = main_bank()
    items = bank["items"]
    print("original bank", bank["meta"]["model"], "items", len(items), "sites", sum(len(it["sites"]) for it in items))

    if mode == "check":
        tok = AutoTokenizer.from_pretrained(ORIG_ID)
        bad = [it["id"] for it in items if sites_for(tok, cap[it["id"]]["input_ids"], cap[it["id"]]["prompt_len"]) != it["sites"]
               or [s["pos"] for s in it["sites"]] != cap[it["id"]]["read_positions"]]
        print("items whose pos/token/kind/char all match the original:", len(items) - len(bad), "of", len(items), "mismatches", bad[:10])
        assert not bad

    if mode == "build":
        from vllm import LLM, SamplingParams

        tok = AutoTokenizer.from_pretrained(SUBJECT_ID)
        prompts = [tok(render(tok, it["prompt"]), add_special_tokens=False).input_ids for it in items]
        same = sum(p == cap[it["id"]]["input_ids"][:cap[it["id"]]["prompt_len"]] for p, it in zip(prompts, items))
        print("prompt renders token-identical to the original bank:", same, "of", len(items))
        print("render example:", repr(tok.decode(prompts[0])))
        stop = {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}
        llm = LLM(SUBJECT_ID, seed=SEED, generation_config="vllm", max_model_len=2048)
        sp = SamplingParams(temperature=TEMP, top_p=TOP_P, top_k=0, max_tokens=MAX_NEW, seed=SEED)
        outs = llm.generate([{"prompt_token_ids": p} for p in prompts], sp)
        new_items, rows, dropped = [], [], []
        for it, p, o in zip(items, prompts, outs):
            gen = list(o.outputs[0].token_ids)
            while gen and gen[-1] in stop:  # EOS not kept
                gen.pop()
            ids, pl = p + gen, len(p)
            sites = sites_for(tok, ids, pl)
            if not sites:
                dropped.append(it["id"])
                continue
            new_items.append({"id": it["id"], "stratum": it["stratum"], "source": it["source"], "prompt": it["prompt"],
                              "response": tok.decode(gen), "response_start": pl, "sites": sites})
            rows.append({"id": it["id"], "source": it["source"], "source_id": cap[it["id"]]["source_id"], "prompt_len": pl,
                         "input_ids": ids, "read_positions": [s["pos"] for s in sites]})
        lens = [len(r["input_ids"]) - r["prompt_len"] for r in rows]
        print("items", len(new_items), "dropped (no sites)", dropped, "sites", sum(len(it["sites"]) for it in new_items),
              "response tokens min/median/max", min(lens), sorted(lens)[len(lens) // 2], max(lens), "hit max_new", sum(n >= MAX_NEW for n in lens))
        print("site kinds", {k: sum(s["kind"] == k for it in new_items for s in it["sites"]) for k in SITE_KINDS})
        assert all(it["response"][:s["char"]].endswith(s["token"]) for it in new_items for s in it["sites"])
        print("example response:", repr(new_items[0]["response"][:300]))
        meta = dict(bank["meta"], model=SUBJECT_ID, n_items=len(new_items), n_sites=sum(len(it["sites"]) for it in new_items),
                    dropped_no_sites=bank["meta"]["dropped_no_sites"] + dropped,
                    rollouts=f"{SUBJECT_ID} on-policy, vLLM generate, T={TEMP}, top-p {TOP_P}, top-k 0, max {MAX_NEW} new tokens, seed {SEED} "
                             "(per request); chat template with an empty think block; EOS not kept. Prompts = the original bank's 149.",
                    read_sites="scripts/port/hallucination.py response_sites/select_sites (reimplementation of hallucination_bench.sites; "
                               "reproduces all 1,123 original sites exactly, `check` mode), thinned to 8 per item",
                    ported_from=f"{ORIG_ID} bank on main (prompts, ids, sources)")
        (OUT / "items.json").write_text(json.dumps({"meta": meta, "items": new_items}, ensure_ascii=False, indent=1) + "\n")
        (OUT / "capture_rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n")
        print("wrote", OUT / "items.json", OUT / "capture_rows.json")
        pj = BENCH / "evals/PORT.json"
        port = json.loads(pj.read_text())
        port["deferred"].pop("hallucination", None)
        port["families"]["hallucination"] = {"rule": "rebuilt by scripts/port/hallucination.py: same prompts, subject's own on-policy responses, sites re-derived", "kept": len(new_items)}
        pj.write_text(json.dumps(port, ensure_ascii=False, indent=1) + "\n")
        print("PORT.json updated: hallucination kept", len(new_items))


if __name__ == "__main__":  # vLLM spawns workers
    main()
