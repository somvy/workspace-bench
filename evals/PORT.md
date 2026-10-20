# Qwen3-8B port (branch `port-qwen3-8b`)

> **Not ported, emptied on this branch:** agentic_misalignment, jlens_concept_pr (their banks are Qwen3.6-27B rollouts and
> activations), brew_intermediates (8 items pass; its read positions are Qwen3.6-27B token indices, not remapped).
> **Below 20 items, flagged in PORT.json:** chain_intermediates 15, conjunctive_association 12, relational_multihop 11,
> multilingual_mt 18, multilingual_multihop 16. **No R-lens exists for this model:** `method=rlens` fails at load. The
> J-lens default is neuronpedia's Qwen3-8B wikitext fit (n=1000), not yet run on the bench. **Judging** goes through
> OpenRouter (Gemini 3.8 Flash) by default, or a local server with `WSBENCH_BASE_URL` (README, "Local judge").

Built like the Qwen2.5-7B port (branch `port-qwen2.5-7b`). Subject: `Qwen/Qwen3-8B` (post-trained, thinking off), 36 layers.
Qwen3-8B sits between the two earlier ports:
- **Same chat template text as Qwen3.6-27B.** No default system prompt; thinking off renders `<think>\n\n</think>\n\n`
  after the assistant header. Every render decodes to the same text as on the 27B.
- **Different tokenizer.** It has Qwen2.5's vocabulary (151,669 tokens vs 248,077), so token ids and word splits differ,
  and every tokenizer-dependent step of the 7B port is redone here.

`PORT.json` has per-family counts. Build scripts: `scripts/port/` (README there).

## Gate (chat only)
Same rule as the 9B and 7B ports: the bench's `capable` question in chat format (bench ANSWER_SYSTEM, thinking off),
10 samples at T=0.7 (top_p=1, top_k=0) plus greedy, graded by local Qwen3.6-27B with the bench GRADE prompt. An item
passes if the greedy answer is right and at least 8/10 samples are right (10/10 for chain and brew); a multihop item needs
every bridge question to pass too. Every answer was graded (0 undecided). Kept: user_modeling 93, multilingual 87,
typo 82, multihop 73, basic_readout 66 (+19 implicit), typo_mt 62, basic_readout_mt 47, association 37,
multilingual_typo 36, buggy_code 29, multihop_mt 28, role_bound_association 22, arithmetic 180. Total over the families
with at least 20 items: 1,485.

Compared with the 7B under the same gate, the 8B is weaker on chain (15 vs 57 kept), relational_multihop (11 vs 38) and
association (37 vs 50), and stronger on role_bound_association (22 vs 2), multilingual_mt (18 vs 7) and buggy_code
(29 vs 23).

## Rebuilt families
- **poetry**: kept if the 8B commits to a rhyme (chat greedy, at least 8/10 the same word, and the plain-render greedy
  continuation agrees). 68/100 kept (20 plain continuation differs, 12 chat inconsistent); 65 of 68 rhymes equal Qwen3.6's.
- **basic_readout implicit**: gold = the 8B's own favourite (chat_prefill render, at least 8/10 the same answer).
  19/32 kept, 12 the same as the bank's.
- **directed_modulation**: compliance re-screened with free generation (copies the carrier verbatim, never names the
  concept; greedy + at least 8/10; pairs kept whole). 97 items pass, 94 kept as whole pairs.
- **moral_rationale**: side split = the 8B's 10 chat answers; reasons = Qwen3.6-27B clustering of the 8B's own
  explanations (same pipeline as the 9B). 200/200 kept: 187 committed, 13 deliberative (bank 166 / 34). The 8B is more
  one-sided than the 27B, the opposite of the 9B (70 deliberative). Same direction as the bank on 140/200 items, same
  class on 167. Variant "mixed" (shipped): 128 items keep the bank's reasons, 72 rebuilt. The 8B's top committed reason is
  judged the same as the bank's look_for_reasons[0] on 51/121 (the fidelity control, Qwen3.6-27B on its own bank: 77/135).
  The blind lucky-guessing floor was not re-measured.
- **hallucination**: same 149 prompts, the 8B's own on-policy responses (vLLM, T=1.0, top-p 1.0, top-k 0, 512 new tokens,
  seed 7, empty think block, EOS dropped; FlashInfer sampler off, which does not change the distribution), read sites
  re-derived on its tokens by `scripts/port/hallucination.py` (`check` reproduces all 1,123 original sites, 149/149 items).
  149 items, 1,029 sites (clause 370, sentence 335, markup 176, newline 131, quote 17); median response 114 tokens vs 228
  on the 27B; 36 hit the 512 cap (original 35). Prompt renders are the same text as the original, different token ids.

## Read positions (tokenizer-dependent)
`scripts/port/poscheck.py` renders every plan row with both tokenizers, resolves its positions rule under each and
compares the selected text (counts below are on main's banks, before the gate).
- **Text rules carry over.** Where the selected text differs, Qwen3-8B splits the last word differently (`' milion'` vs
  `'ion'`): basic_readout_mt 14, typo_mt 5, typo 2, multilingual 1, and many non-Latin multilingual_mt (53),
  multilingual_multihop (33) and multilingual_typo (19) items. Poetry's line-one newline is `',\n'` (one token) instead
  of `'\n'`, as on Qwen2.5.
- **Chat tails are the same text as on the 27B**, so the chat families (moral_rationale, user_modeling,
  conjunctive/role_bound summarize tails, chain, arithmetic) select identical text: 0 rows differ. moral_rationale's
  5 tail positions are `\n<think>\n\n</think>\n\n`, as on the 27B (on Qwen2.5 they were the header tokens).
- **jailbreak_recognition** stored token indices; `scripts/port/remap.py` re-derives each read (every token of the last
  user turn through its `<|im_end|>`) on Qwen3-8B's render. `remap.py check` rebuilds all 86 original reads exactly under
  the Qwen3.6-27B tokenizer. Span tokens total 28,458 (bank 28,713).
- **arithmetic_intermediates** frozen cells -8 / -7 (the newline after `<|im_end|>`, `<|im_start|>`) map to the same
  offsets -8 / -7: the tail tokenizes identically. They were the best cells on the 27B, so on the 8B they are still a
  guess: use `opts=cells=all` until they are re-selected.
- **brew_intermediates** stores token indices too and is not remapped; with 8 passing items it is emptied.
- **The six multi-token families** are restamped with Qwen3-8B's tokenizer (`scripts/port/restamp.py`; `check`
  reproduces all 4,600 original stamps). Counts mostly go up; 16 forms become creditable (multilingual_mt 8,
  multilingual_typo 8), none lose credit, no item is dropped.

## Layers
Every layer l becomes min(int(l * 36 / 64 + 0.5), 35) (64 to 36 layers): GRID [11, 14, 16, 18, 20, 23, 25, 27, 29, 32, 34],
SIX [11, 16, 20, 25, 29, 34], FIVE [11, 20, 25, 29, 34]; arithmetic 32/34; buggy_code 34/32; MCDM [25, 29, 32, 34];
agentic [*GRID, 35]; producer default and jlens_concept_pr headline 25.

## Checks run
- `scripts/port/fullcheck.sh` on Qwen3-8B: `wsbench produce method=logit_lens` for all 27 families (exit 0), then
  `wsbench run all=True dry_run=True` (exit 0): every family reaches its judge with the expected item count, the six
  regex families score with 0 missing cells, hallucination expects 5,145 cells (1,029 sites x 5 layers), 0 missing.
  No judge call was made; no lens has been scored on this branch.
- `remap.py check`, `restamp.py check` and `hallucination.py check` reproduce the original bank exactly (86/86 reads,
  all multi-token stamps, 1,123/1,123 sites). `assemble.py` + `remap.py write` + `restamp.py write` re-run from scratch
  give byte-identical banks.
- Tests: the code tests pass; 131 tests that pin the original banks (item counts, golden prompts, bank positions/layers,
  multi-token stamps) fail, as on the 7B branch (135 there). With main's banks swapped in, this branch's code fails 5
  tests, all pinning 27B layer numbers (the same 5 as on the 7B branch).

## Not ported
- agentic_misalignment (needs 8B rollouts), jlens_concept_pr (needs 8B rollouts and activations; source prompts are private).
- The NLA method still points at the 27B NLA. A Qwen3-8B NLA exists on the Hub (`asher577/nla-qwen-3-8b`) but is not wired in.
