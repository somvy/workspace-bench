# render every bench read-plan row with the Qwen3.6-27B and Qwen3.5-9B tokenizers; count rows whose ids differ
from collections import Counter
from transformers import AutoTokenizer
from wsbench.readplan import plan
from wsbench.registry import FAMILIES, load_all
load_all()
from wsbench.produce.render import render
A, B = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B"), AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
print("vocab", len(A), len(B), "same vocab", A.get_vocab() == B.get_vocab())
for f in FAMILIES:
    try:
        rows = plan(f)
    except Exception as e:
        print(f, "plan error", repr(e)[:120]); continue
    renders, diff, ex = Counter(r.render for r in rows), 0, None
    for r in [r for r in rows if r.render != "captured" or "input_ids" in r.extra]:
        a, b = render(r, A).ids, render(r, B).ids
        if a != b:
            diff += 1
            ex = ex or (r.id, len(a), len(b), next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None))
    print(f, dict(renders), "rows", len(rows), "differ", diff, "example", ex)
