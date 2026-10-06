# Qwen2.5-7B-Instruct port (branch `port-qwen2.5-7b`)

> **Not ported, emptied on this branch:** agentic_misalignment, jlens_concept_pr (their banks are Qwen3.6-27B rollouts and
> activations). **Below 20 items:** multilingual_mt 7, multilingual_multihop 13, conjunctive_association 14,
> brew_intermediates 0; role_bound_association kept 2 and is emptied (its judge draws 3 distractor scenes from other
> items). **No J-lens / R-lens exists for this model:** `method=jlens|rlens` fails at load. **Judging** goes through
> OpenRouter (Gemini 3.8 Flash) by default, or a local server with `WSBENCH_BASE_URL` (README, "Local judge").

Built like the Qwen3.5-9B port (branch `port-qwen3.5-9b`), with one difference that matters: Qwen2.5 has a different
tokenizer (151,665 vs 248,077 tokens) and chat template (no think block; the template inserts its default system prompt,
"You are Qwen, created by Alibaba Cloud. You are a helpful assistant.", when an item has none, and it is kept, as the
model is normally run). Every chat render on this branch is Qwen2.5's own template. `PORT.json` has per-family counts.
Build scripts: `scripts/port/` (README there).

## Gate (chat only)
Same rule as the 9B port: the bench's `capable` question in chat format (bench ANSWER_SYSTEM), 10 samples at T=0.7
(top_p=1, top_k=0) plus greedy, graded by local Qwen3.6-27B with the bench GRADE prompt. An item passes if the greedy
answer is right and at least 8/10 samples are right (10/10 for chain and brew); a multihop item needs every bridge
question to pass too. Kept: typo 94, user_modeling 94, multilingual 90, multihop 79, basic_readout 67 (+22 implicit),
typo_mt 60, chain 57, association 50, relational_multihop 38, basic_readout_mt 37, multilingual_typo 33, buggy_code 23,
multihop_mt 21, arithmetic 194. Total over all families: 1,579 items.

## Rebuilt families
- **poetry**: kept if the 7B commits to a rhyme (chat greedy, at least 8/10 the same word, and the plain-render greedy
  continuation agrees). 71/100 kept (15 plain continuation differs, 14 chat inconsistent); 69 of 71 rhymes equal Qwen3.6's.
- **basic_readout implicit**: gold = the 7B's own favourite (chat_prefill render, at least 8/10 the same answer).
  22/32 kept, 9 the same as the bank's.
- **directed_modulation**: compliance re-screened with free generation (copies the carrier verbatim, never names the
  concept; greedy + at least 8/10; pairs kept whole). 94 items pass, 88 kept as whole pairs.
- **moral_rationale**: side split = the 7B's 10 chat answers; reasons = Qwen3.6-27B clustering of the 7B's own
  explanations (same pipeline as the 9B). 199/200 kept: 177 committed, 22 deliberative (bank 166 / 34). Same direction as
  the bank on 139/199 items, same class on 153. Variant "mixed" (shipped): 121 items keep the bank's reasons, 78 rebuilt.
  The 7B's top committed reason is judged the same as the bank's look_for_reasons[0] on 49/116 (Qwen3.6-27B on its own
  bank, the fidelity control from the 9B port: 77/135). The blind lucky-guessing floor was not re-measured.
- **hallucination**: same 149 prompts, the 7B's own on-policy responses (vLLM, T=1.0, top-p 1.0, top-k 0, 512 new tokens,
  seed 7, EOS dropped), read sites re-derived on its tokens by `scripts/port/hallucination.py` (whose `check` mode
  reproduces all 1,123 original sites). 149 items, 1,096 sites (clause 458, sentence 362, newline 142, markup 109,
  quote 25); responses are shorter than Qwen3.6-27B's (median 126 tokens vs 228; 26 hit the 512 cap vs 35).

## Read positions (tokenizer-dependent)
`scripts/port/poscheck.py` renders every plan row with both tokenizers, resolves its positions rule under each and
compares the selected text.
- **Text rules carry over** (final_token, offset_from_end, line_one_newline, from_token, suffix_text, last_n,
  from_last_sentence_start). Where the selected text differs it is because Qwen2.5 splits a word differently: the final
  token of 6 basic_readout_mt, 4 typo_mt, 2 typo, 1 multilingual and most Arabic/Hebrew/Greek multilingual items is a
  different piece of the same last word (`' milion'` vs `'ion'`); poetry's line-one newline is `',\n'` (one token in
  Qwen2.5) instead of `'\n'`. The chat families read the same user text followed by Qwen2.5's own, 4-token-shorter tail.
- **moral_rationale** reads the last 5 positions, "the assistant-header tail". On Qwen3.6 those are `\n<think>\n\n</think>\n\n`;
  on Qwen2.5 they are `<|im_end|>\n<|im_start|>assistant\n`. The rule is unchanged; what it reads differs by template.
- **jailbreak_recognition** stored token indices; `scripts/port/remap.py` re-derives each item's read (every token of the
  last user turn through its `<|im_end|>`) on Qwen2.5's render. The same code rebuilds all 86 original reads exactly
  under the Qwen3.6-27B tokenizer (`remap.py check`); on Qwen2.5 every span equals the last user turn + `<|im_end|>`, 86/86.
- **arithmetic_intermediates** frozen cells were offsets in the chat tail (-8 = the newline after `<|im_end|>`,
  -7 = `<|im_start|>`); mapped by token identity to -4 / -3. They were the best cells on the 27B, so on the 7B they are
  a guess: use `opts=cells=all` until they are re-selected.
- **hallucination** is captured token ids, rebuilt above.
- **The six multi-token families** (typo_mt, multihop_mt, multilingual_mt, basic_readout_mt, multilingual_multihop,
  multilingual_typo) credit an answer form only when it is strictly multi-token under the probed model's tokenizer
  (`probe_token_lens` > 1), so a top-k token lens cannot hit it with one token. The counts are per-tokenizer;
  `scripts/port/restamp.py` restamps them with Qwen2.5's tokenizer. The rule (minimum token count over as-is / lower /
  title case, bare and with a leading space) reproduces all 4,600 original Qwen3.6-27B stamps exactly (`restamp.py check`).
  On Qwen2.5 counts mostly go up (smaller vocabulary): 13 forms become creditable (multilingual_mt 3, multilingual_typo 10),
  none lose credit, no item is dropped.

## Layers
Every layer l becomes min(int(l * 28 / 64 + 0.5), 27) (64 to 28 layers): GRID [9, 11, 12, 14, 16, 18, 19, 21, 23, 25, 26],
SIX [9, 12, 16, 19, 23, 26], FIVE [9, 16, 19, 23, 26]; arithmetic 25/26; buggy_code 26/25; MCDM [19, 23, 25, 26];
agentic [*GRID, 27]; producer default 19.

## Checks run
- `scripts/port/fullcheck.sh` on Qwen2.5-7B-Instruct: `wsbench produce method=logit_lens` for all 27 families (exit 0),
  then `wsbench run all=True dry_run=True` (exit 0): every family reaches its judge with the expected item count, the six
  regex families score with 0 missing cells, hallucination expects 5,480 cells (1,096 sites x 5 layers), 0 missing.
  No judge call was made; no lens has been scored on this branch.
- Tests: the code tests pass; 135 tests that pin the original banks (item counts, golden prompts, bank positions/layers)
  fail, as on the 9B branch (119 there; the extra ones pin jailbreak positions, typo/typo_mt items, 27B layer numbers
  and the multi-token stamps). With main's banks swapped in, this branch's code fails 5 tests, all pinning 27B layer numbers.

## Not ported
- agentic_misalignment (needs 7B rollouts), jlens_concept_pr (needs 7B rollouts and activations; source prompts are private).
- The NLA method still points at the 27B NLA; a released Qwen2.5-7B NLA exists (layer 20) but is not wired in.
