# probe_token_lens for a subject whose tokenizer differs from Qwen3.6-27B (run after assemble.py). The six multi-token
# families credit a form only when it is strictly multi-token under the probed model's tokenizer
# (wsbench.multitoken.regex.scored_units: count > 1). The count is the minimum token length over {as-is, lower, title}
# x {bare, leading space}; `check` recomputes every stamp of main's banks with the Qwen3.6-27B tokenizer and requires an
# exact match, `write` restamps this branch's banks with the subject tokenizer and drops items left with a required
# unit that has no creditable form. Idempotent.
# usage: python restamp.py check | write
import json, subprocess, sys
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer
from wsbench.multitoken.regex import contract_for, scored_units

ORIG_ID, SUBJECT_ID = "Qwen/Qwen3.6-27B", "Qwen/Qwen2.5-7B-Instruct"
BENCH = Path(__file__).resolve().parents[2]
FAMS = ["typo_mt", "multihop_mt", "multilingual_mt", "basic_readout_mt", "multilingual_multihop", "multilingual_typo"]


def count(tok, s):
    return min(len(tok(p + v, add_special_tokens=False)["input_ids"]) for v in {s, s.lower(), s.title()} for p in ("", " "))


def stamp(tok, it):
    units = {}
    for u in it["units"]:
        forms = u.get("forms") or {"": u.get("match", [])}
        units[u["role"]] = {lang: [count(tok, f) for f in fs] for lang, fs in forms.items()}
    out = {**it["probe_token_lens"], "units": units}
    if "target" in out:
        out["target"] = count(tok, it["target"])
    if "target_alts" in out:
        out["target_alts"] = [count(tok, f) for f in it["target_alts"]]
    return out


def creditable(it):  # (role, lang, form) of every strictly multi-token form of the units the count filters
    out = set()
    for u in it["units"]:
        if not u.get("multi_token", True):
            continue
        forms = u.get("forms") or {"": u.get("match", [])}
        for lang, fs in forms.items():
            out |= {(u["role"], lang, f) for f, n in zip(fs, it["probe_token_lens"]["units"][u["role"]][lang]) if n > 1}
    return out


def run():
    mode = sys.argv[1]
    if mode == "check":
        A = AutoTokenizer.from_pretrained(ORIG_ID)
        for f in FAMS:
            bank = json.loads(subprocess.run(["git", "-C", str(BENCH), "show", f"main:evals/{f}/items.json"], capture_output=True, check=True).stdout)
            bad = [it["id"] for it in bank["items"] if stamp(A, it) != it["probe_token_lens"]]
            print(f, "stamps rebuilt with", ORIG_ID, "equal to the bank:", len(bank["items"]) - len(bad), "of", len(bank["items"]), "mismatches", bad[:5])
            assert not bad
        return
    B = AutoTokenizer.from_pretrained(SUBJECT_ID)
    pj = BENCH / "evals/PORT.json"
    port = json.loads(pj.read_text())
    for f in FAMS:
        path = BENCH / "evals" / f / "items.json"
        bank = json.loads(path.read_text())
        contract = contract_for(bank, bank["family"])
        kept, dropped, flips, ups = [], [], Counter(), Counter()
        for it in bank["items"]:
            new = {**it, "probe_token_lens": stamp(B, it)}
            before, after = creditable(it), creditable(new)
            flips["gained"] += len(after - before)
            flips["lost"] += len(before - after)
            for role, langs in new["probe_token_lens"]["units"].items():
                for lang, ns in langs.items():
                    for a, b in zip(it["probe_token_lens"]["units"][role][lang], ns):
                        ups["up" if b > a else "down" if b < a else "same"] += 1
            try:
                scored_units(new, contract)
            except ValueError as e:
                dropped.append((it["id"], str(e)))
                continue
            kept.append(new)
        print(f, "items", len(bank["items"]), "kept", len(kept), "dropped", dropped, "| creditable forms", dict(flips), "| unit counts vs bank", dict(ups))
        path.write_text(json.dumps({**bank, "items": kept}, ensure_ascii=False, indent=1) + "\n")
        fam = port["families"][f]
        fam["kept"] = len(kept)
        fam["probe_token_lens"] = f"restamped with {SUBJECT_ID} (scripts/port/restamp.py); {dict(flips)} creditable forms vs the bank"
        if dropped:
            fam["restamp_dropped"] = [i for i, _ in dropped]
    pj.write_text(json.dumps(port, ensure_ascii=False, indent=1) + "\n")
    print("wrote the six banks and PORT.json")


if __name__ == "__main__":
    run()
