# Qwen3.5-9B port (branch `port-qwen3.5-9b`)

> **Not ported, emptied on this branch:** hallucination, agentic_misalignment, jlens_concept_pr. Their banks are
> Qwen3.6-27B token ids and rollouts. **Below 20 items:** brew 0, relational_multihop 18, conjunctive_association 10.
> **Judging** of every non-regex family, and the token-lens summarizer, still go through OpenRouter (Gemini 3.8 Flash).

Every bank on this branch is the Qwen3.6-27B bank from `main`, filtered to the items `Qwen/Qwen3.5-9B` (post-trained) passes,
with the model-specific golds rebuilt. `PORT.json` has per-family counts and rules. Build scripts: `scripts/port/` (README there).

## Gate (chat only)
The bench's `capable` question in chat format (bench ANSWER_SYSTEM, thinking off), 10 samples at T=0.7 (top_p=1, top_k=0) plus
greedy. Each answer is graded by local Qwen3.6-27B with the bench GRADE prompt. An item passes if the greedy answer is right
and at least 8/10 samples are right (10/10 for chain and brew). A multihop item needs its surface question AND every bridge
question to pass. Sampling with no top_p/top_k cut-off is stricter than the banks' own gate, so the kept items are a
conservative subset.

## Rebuilt families
- **poetry** (re-gated; no gold changed): kept if the 9B commits to a rhyme (chat greedy, at least 8/10 samples give the same
  word, and the plain-render greedy continuation agrees). 76/100 kept, and every kept rhyme equals Qwen3.6's.
- **basic_readout implicit**: gold = the 9B's own favourite (chat_prefill render, at least 8/10 the same answer; "none" etc.
  excluded). Only 3/32 kept: the 9B mostly answers "None".
- **directed_modulation**: compliance re-screened with free generation (copies the carrier verbatim, never names the concept;
  greedy + at least 8/10; pairs kept whole). 100/100.
- **moral_rationale**: side split = the 9B's 10 chat answers. Reasons = Qwen3.6-27B clustering of the 9B's own explanations,
  one follow-up "main reason" turn per sample, conditioned on that sample's answer.
  - Committed: greedy = majority and majority at least 8/10. Deliberative: minority at least 2/10 and reasons on both sides.
  - The 9B is more split than the 27B: 70 deliberative vs 34 in the bank. The same rule on the 27B gives 38, so this is the
    9B's behaviour, not the rule. Deliberative items need both sides read, so this family is harder on the 9B by composition.
  - Variant "mixed" (shipped): where the 9B agrees with the bank (same class, and same side if committed), the bank's
    reasons are kept (117 items); the other 82 are rebuilt. The all-rebuilt variant can be produced by setting `MORAL_VARIANT = "rebuilt"` in `scripts/port/assemble.py`.
  - Fidelity control (same pipeline with Qwen3.6-27B as the model being read): side agrees with the bank on 194/199 items,
    class on 143/199, and the top reason is judged the same as the bank's look_for_reasons[0] on 77/135. The rebuilt reasons
    come from a related but not identical instrument; about 25% of items tie for the top reason.
  - Blind lucky-guessing floor (Qwen3.6-27B guesser, 5 draws T=1): original 0.193 (bench's frozen Gemini number 0.195),
    mixed 0.171, rebuilt 0.177; chance 0.20.

## Unchanged
multi_concept_directed_modulation and jailbreak_recognition have no model gate. `tplcheck.py`: Qwen3.5-9B has the same
tokenizer and renders every read render of every family to identical token ids, so their positions carry over.

## Layers
Every layer l becomes l // 2 (64 to 32 layers): GRID 10..30 step 2, SIX/FIVE likewise; arithmetic 28/30 and its frozen
cells (the frozen cells were the best cell per variant on the 27B; on the 9B the mapped cell is a guess, so use
`opts=cells=all` until they are re-selected); buggy_code 30/28; MCDM [22,26,28,30]; agentic 31; producer default 22.
Layer 30 is the 9B J/R-lens target layer: J = identity there, so those lenses become logit-lens-like at 30.
JLens/RLens default to `camilablank/workspace-lenses` `qwen3.5-9b/{j,r}-lens/lens.pt` (n=25). No n=1000 J-lens exists for 9B.
Smoke test (typo_mt): with the cosine readout the J-lens peaks at layer 10 and falls to 0 at layers 24-28. Check it before trusting it.

## Checks run
- `scripts/port/fullcheck.sh` on Qwen3.5-9B: `wsbench produce method=logit_lens` for all 27 families (exit 0), then
  `wsbench run all=True dry_run=True`. Every family reaches its judge with the expected item count; the six regex families
  score with 0 missing cells. No judge/summarizer API call was made, so no lens score on the LLM-judged families exists yet.

## Dropped / not ported
- Fewer than 20 items, left in place but flagged in PORT.json: brew 0, relational_multihop 18, conjunctive_association 10.
- Not ported, item lists emptied: hallucination (needs the 9B's own responses + labels), agentic_misalignment (needs 9B
  rollouts), jlens_concept_pr (needs 9B rollouts and activations; source prompts are private).
- The NLA method still points at the 27B NLA; no 9B NLA exists.
- `tests/golden` pins the original banks and will fail on this branch.
