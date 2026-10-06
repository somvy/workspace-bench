# For a subject with a DIFFERENT tokenizer/template than Qwen3.6-27B: render every read-plan row with both tokenizers,
# resolve its positions rule under each, and compare the text the selected tokens decode to. A row "differs" when the
# subject reads different text than the bank's model did. Run before assemble (reads the plan of the current checkout).
# usage: python poscheck.py [family ...]
import sys
from collections import Counter
from transformers import AutoTokenizer
from wsbench.readplan import plan, resolve
from wsbench.registry import FAMILIES, load_all

ORIG_ID, SUBJECT_ID = "Qwen/Qwen3.6-27B", "Qwen/Qwen2.5-7B-Instruct"
load_all()
from wsbench.produce.render import render

A, B = AutoTokenizer.from_pretrained(ORIG_ID), AutoTokenizer.from_pretrained(SUBJECT_ID)
print("vocab", len(A), len(B))


def picked(r, tok):
    x = render(r, tok)
    pos = resolve(r.positions, x.tokens)
    return x, pos, "".join(x.decoded[p] for p in pos)


fams = sys.argv[1:] or sorted(FAMILIES)
for f in fams:
    rows = [r for r in plan(f) if r.render != "captured"]
    if not rows:
        print(f, "no renderable rows"); continue
    diff, nlen, ex = 0, Counter(), []
    for r in rows:
        (xa, pa, ta), (xb, pb, tb) = picked(r, A), picked(r, B)
        nlen[(len(pa), len(pb)) if len(pa) != len(pb) else "same n"] += 1
        if ta != tb:
            diff += 1
            if len(ex) < 2:
                ex.append((r.id, r.positions, repr(ta[-80:]), repr(tb[-80:])))
    r0 = rows[0]
    print(f"{f}: rows {len(rows)} renders {dict(Counter(r.render for r in rows))} rule {r0.positions['kind']} | selected text differs {diff} | n positions {dict(nlen.most_common(3))}")
    print("   render tail A", repr(A.decode(render(r0, A).ids)[-90:]))
    print("   render tail B", repr(B.decode(render(r0, B).ids)[-90:]))
    for e in ex:
        print("   DIFF", e)
