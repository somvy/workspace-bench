# Tokenizer-dependent positions for a subject whose tokenizer/chat template differ from Qwen3.6-27B (run after assemble.py).
# jailbreak_recognition: read = every token of the last user turn, first content token through its <|im_end|>, re-derived
#   on the subject's render (+ the legacy 13-site `positions`, `n_tokens`, `tokens`). `check` rebuilds the bank's reads
#   with the Qwen3.6-27B tokenizer and requires an exact match.
# arithmetic_intermediates: frozen cells are offsets from the end of the chat render that land in the template tail
#   (-8 = the newline after <|im_end|>, -7 = <|im_start|>); each maps to the subject's token of the same identity.
# usage: python remap.py check | write
import json, subprocess, sys
from pathlib import Path
from transformers import AutoTokenizer
from wsbench.evals.jailbreak_recognition.judge import prefix_to_last_user
from wsbench.produce.render import _chat

ORIG_ID, SUBJECT_ID = "Qwen/Qwen3.6-27B", "Qwen/Qwen2.5-7B-Instruct"
BENCH = Path(__file__).resolve().parents[2]
N_GRID = 12  # legacy read.positions: up to 12 evenly spaced content tokens + turn_end


def main_items(family):
    return json.loads(subprocess.run(["git", "-C", str(BENCH), "show", f"main:evals/{family}/items.json"], capture_output=True, check=True).stdout)


def jb_read(tok, messages):
    ids = _chat(tok, prefix_to_last_user(messages))
    dec = [tok.decode([i]) for i in ids]
    starts = [i for i in range(len(dec) - 2) if dec[i] == "<|im_start|>" and dec[i + 1] == "user" and dec[i + 2] == "\n"]
    a = starts[-1] + 3
    b = next(i for i in range(a, len(dec)) if dec[i] == "<|im_end|>")
    grid = list(dict.fromkeys([round(a + k * (b - 1 - a) / (N_GRID - 1)) for k in range(N_GRID)] if b - 1 > a else [a]))
    return {"positions": [p for p in grid if p < b] + [b], "turn_end": b, "n_tokens": len(ids) + 2, "span": [a, b],
            "tokens": {str(p): dec[p] for p in range(a, b + 1)}}


def tail_map(ta, tb, prompt, offsets):
    # offset in A's chat render -> offset in B's, by token identity counted from the last <|im_end|>
    def tail(tok):
        ids = _chat(tok, [{"role": "user", "content": prompt}])
        dec = [tok.decode([i]) for i in ids]
        e = max(i for i, d in enumerate(dec) if d == "<|im_end|>")
        return dec, e, len(ids)
    (da, ea, na), (db, eb, nb) = tail(ta), tail(tb)
    out = {}
    for off in offsets:
        rel = na + off - ea  # tokens after A's last <|im_end|>
        assert 0 <= rel < nb - eb and da[ea + rel] == db[eb + rel], (off, da[ea + rel:], db[eb:])
        out[off] = eb + rel - nb
        print("arith cell", off, "A token", repr(da[na + off]), "-> B offset", out[off], "B token", repr(db[nb + out[off]]))
    return out


def run():
    mode = sys.argv[1]
    A = AutoTokenizer.from_pretrained(ORIG_ID)
    jb = main_items("jailbreak_recognition")
    if mode == "check":
        bad = [it["id"] for it in jb["items"] if jb_read(A, it["messages"]) != it["read"]]
        print("jailbreak reads rebuilt with", ORIG_ID, "equal to the bank:", len(jb["items"]) - len(bad), "of", len(jb["items"]), "mismatches", bad[:5])
        assert not bad
        return
    B = AutoTokenizer.from_pretrained(SUBJECT_ID)
    new = [{**it, "read": jb_read(B, it["messages"])} for it in jb["items"]]
    spans = [it["read"]["span"][1] - it["read"]["span"][0] for it in new]
    print("jailbreak items", len(new), "span tokens total", sum(spans), "min/max", min(spans), max(spans),
          "(bank:", sum(it["read"]["span"][1] - it["read"]["span"][0] for it in jb["items"]), ")")
    r0 = new[0]["read"]
    print("example span ends:", repr(r0["tokens"][str(r0["span"][0])]), "...", repr(r0["tokens"][str(r0["span"][1])]))
    (BENCH / "evals/jailbreak_recognition/items.json").write_text(json.dumps({**jb, "items": new}, ensure_ascii=False, indent=1) + "\n")
    path = BENCH / "evals/arithmetic_intermediates/items.json"
    ar = json.loads(path.read_text())
    offs = sorted({v["cell"]["pos"] for v in ar["variants"].values()} | {it["cell"]["pos"] for it in ar["items"]})
    m = tail_map(A, B, ar["items"][0]["prompt"], offs)
    ar["variants"] = {k: {**v, "cell": {**v["cell"], "pos": m[v["cell"]["pos"]]}} for k, v in ar["variants"].items()}
    ar["items"] = [{**it, "cell": {**it["cell"], "pos": m[it["cell"]["pos"]]}} for it in ar["items"]]
    path.write_text(json.dumps(ar, ensure_ascii=False, indent=1) + "\n")
    print("wrote jailbreak_recognition reads and arithmetic cell offsets", m)


if __name__ == "__main__":
    run()
