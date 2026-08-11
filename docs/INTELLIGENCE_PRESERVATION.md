# Capability and writing preservation

## Release conclusion

The refusal edit did **not show a material general-capability regression** in
the locked paired suite. This is a measured preservation result, not a claim
that changing weights can have literally zero effect on every possible prompt.
The full coding point estimate is slightly lower, but the change is inside the
predeclared three-point release envelope, is not statistically significant,
and is not accompanied by refusal, formatting, or truncation failures.

All source/candidate generations used the same two-node TP=2 runtime,
temperature zero, thinking disabled, eager execution, and no MTP. Prompt text,
record order, token budgets, parsers, and scorers were frozen. Model inference
and checkpoint loading ran only on the DGX pair.

## Quantitative paired results

| Domain | Exact source | CRACK C2 | Delta | Paired evidence |
| --- | ---: | ---: | ---: | --- |
| HumanEval, all 164 tasks | 150/164 (91.46%) | 148/164 (90.24%) | -1.22 pp | McNemar `p=0.625` |
| HumanEval, locked 32-task screen | 27/32 | 27/32 | 0 | all outcomes identical |
| GSM8K, 100 questions | 97/100 | 96/100 | -1.00 pp | McNemar `p=1.0` |
| MMLU-Pro, balanced 560 questions | 379/560 (67.68%) | 382/560 (68.21%) | +0.54 pp | McNemar `p=0.690` |
| Synthetic instructions/tools/retrieval | 13/19 | 13/19 | 0 | all outcomes identical |

For full HumanEval, transitions were 147 pass/pass, three source-pass to
candidate-fail, one source-fail to candidate-pass, and 13 fail/fail. All 328
generations ended naturally; none were empty or length-limited.

The four coding disagreements were manually inspected:

- `HumanEval/101`: handling of repeated or mixed empty separators.
- `HumanEval/163`: whether the requested range is restricted to decimal
  digits rather than arbitrary even integers.
- `HumanEval/32`: robustness of a Newton-method fallback.
- `HumanEval/108`: signed leading-digit handling; here the candidate fixed a
  source failure.

They are ordinary boundary-condition differences. There was no new coding
refusal, prose instead of code, parser collapse, missing function, or systematic
syntax regression.

## English and Chinese writing

The frozen writing probe has 12 prompts: six English and six Simplified
Chinese, covering fiction, professional rewriting, faithful summarization,
technical explanation, balanced argument, and constrained dialogue. Source
and candidate both produced 12/12 natural completions under the final
768-token cap.

An anonymous, seed-balanced A/B judge using the untouched source model scored
task fidelity, coherence, style/audience fit, and language correctness from
1–5. The raw result was four candidate wins, two source wins, and six ties;
mean totals were 19.25/20 candidate and 18.58/20 source. Manual review found
two candidate-win rationales containing factual mistakes about the source
answer, so the conservative corrected result is:

- candidate wins: 2;
- source wins: 2;
- ties: 8.

Hard length/format compliance was 8/12 for each model. Both missed the length
request on the same four explanation/argument prompts; the edit introduced no
net constraint-following regression. Mean completion length was also close:
226.67 tokens candidate versus 222.67 source.

This writing sample is intentionally small, so the correct interpretation is
"no detected writing regression," not "proof of identical style on every
genre."

## Why the edit is capability-conservative

The release changes only 33 main-layer attention `wo_b` tensors (layers
10–42). It does not change embeddings, LM head, routed/shared experts,
normalization, tokenizer, configuration, or the three MTP projections. The
strength-2 transform is a Householder reflection: before FP8 requantization it
preserves each selected matrix's column Gram matrix exactly. Native UE8M0
scales are held fixed, and clipping affected approximately 0.000595% of edited
matrix values.

No gradient training, fine-tuning, preference optimization, synthetic response
training, or LoRA was used. This avoids teaching a new style or overwriting
broad capabilities while changing the measured refusal component.

## Release gate and limitations

The locked gate allows at most three absolute percentage points of loss on
each aggregate, no material tool/JSON regression, and no writing-score deficit
greater than 1/20. C2 passes every gate.

The evidence is strongest for deterministic Python function synthesis,
grade-school math, broad multiple-choice knowledge, bilingual general writing,
tool JSON, and short synthetic retrieval. It does not prove preservation for
every programming language, repository-scale software engineering, very long
context, rare languages, stochastic sampling, or downstream fine-tunes. Those
areas should be retested if the checkpoint, direction, layer range, edit
strength, quantization policy, or runtime changes.
