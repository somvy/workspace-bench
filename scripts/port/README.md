# Scripts that built the subject-model ports (see evals/PORT.md; this branch: Qwen2.5-7B-Instruct)

Run in order. Paths are hard-coded to the machines they ran on (`/workspace` = a 2x A100 box with this repo's
`.venv`, `/mnt/...` = the assembly host); edit `OUT` / `BENCH` / `GATE` / `PORT` to reproduce.

1. `gate.py answer q35_9b` then `gate.py grade <shard> <nshards> q35_9b`: the bench's `capable` questions answered
   locally by the subject (10 samples T=0.7 + greedy, thinking off), graded by local Qwen3.6-27B with the bench GRADE prompt.
2. `port.py gen q35_9b` / `port.py extract q35_9b` (and the same for `q36_27b`, the fidelity control): moral sides +
   explanations, favourite items, directed_modulation compliance, poetry plain-render continuations; then reason clustering
   and matching by Qwen3.6-27B. `port.py floor original=...,mixed=...,rebuilt=...`: blind lucky-guessing floor.
3. `tplcheck.py`: token-id equality of every read render under the Qwen3.6-27B and Qwen3.5-9B tokenizers.
4. `assemble.py`: reads the original banks from `main`, writes the filtered/rebuilt banks + `evals/PORT.json`. Idempotent.
5. `fullcheck.sh`: no-API smoke test: logit-lens readouts for every family on the 9B, then `wsbench run all=True dry_run=True`.
6. `hallucination.py check` then `hallucination.py build` (vLLM venv, 1 GPU): reproduces the original bank's read
   sites, then rebuilds `evals/hallucination/` on the subject's own responses and updates `PORT.json`.
7. Subjects with a different tokenizer/template (Qwen2.5-7B): `poscheck.py` (selected text per positions rule under both
   tokenizers), then after `assemble.py`: `remap.py check` (rebuilds the original jailbreak reads exactly) and
   `remap.py write` (jailbreak reads on the subject's render, arithmetic cell offsets by tail-token identity).
   `remap.py write` is not idempotent: run it once on a fresh `assemble.py` output.
   Then `restamp.py check` (rebuilds the original `probe_token_lens` stamps of the six multi-token banks exactly) and
   `restamp.py write` (restamps them with the subject tokenizer, drops items left without a creditable required form).
Note: `port.py gen` reads the banks of the current checkout; run it on the original (main) banks, not on a branch whose
banks are already filtered for another subject.
