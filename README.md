# Fine-tuning LFM2.5-350M for structured NER extraction

A one-day experiment: take Liquid AI's LFM2.5-350M, fine-tune it with
[leap-finetune](https://github.com/Liquid4All/leap-finetune) on CoNLL-2003 named
entity recognition, and measure what changes.

The task is structured extraction, not classification. The model reads a
sentence and must emit a JSON array of `{"text": ..., "type": ...}` objects,
where `type` is one of `PER`, `ORG`, `LOC`, `MISC`. This tests two things at
once: whether the model finds the right entities, and whether it emits parseable
output conforming to a fixed schema.

## Results

200 held-out CoNLL-2003 test sentences, greedy decoding, identical prompt and
scoring across all three runs.

| | Baseline (untuned) | 1k examples | 8k examples |
|---|---|---|---|
| Strict JSON parse | 0.0% | 92.5% | **99.5%** |
| Precision | 0.111 | 0.326 | **0.907** |
| Recall | 0.195 | 0.175 | **0.837** |
| F1 | 0.141 | 0.228 | **0.871** |
| Out-of-schema type labels | 11 | 15 | **0** |

F1 is entity-level micro F1 over exact `(text, type)` matches. An entity counts
only if both the span boundaries and the type are correct.

Published state of the art on the original CoNLL-2003 test set is 94.3 F1
(LUKE, [Yamada et al. 2020](https://arxiv.org/pdf/2010.01057)), using
entity-aware self-attention over a large pretrained encoder. That number
isn't directly comparable to this run: it comes from an encoder-only
sequence-labeling architecture with a CRF/span classifier that optimizes
per-token tag probabilities directly, not a decoder-only LLM generating
free-text JSON. The more relevant comparison is small generative LLMs
producing *structured JSON* output specifically — a 2026 study on generative
NER ([arXiv:2601.17898](https://arxiv.org/html/2601.17898v1)) found that
1B-1.7B models forced into strict JSON output score only 12-33 F1, because
character-offset/schema generation is hard for small decoders in that
format. Against that baseline, 0.871 F1 from a 350M model doing full JSON
output is a strong result for the class — plausibly near the ceiling for
small-LLM JSON-formatted extraction specifically, rather than for CoNLL-2003
NER in general.

## What the runs show

**The untuned model never produced parseable output.** Not one response in 200
was a bare JSON array, despite a system prompt instructing exactly that. Every
response was wrapped in markdown code fences, or emitted a single bare object
instead of an array. Scoring leniently (stripping fences, wrapping bare objects)
recovers 97% of them, which is why both a strict and a lenient parse rate are
reported. The gap between the two is a measure of format compliance
independent of task accuracy.

**Untuned task accuracy was near-random.** The model tagged nearly every
capitalized token as an entity and assigned types with little discrimination:
`SOCCER` as `PER`, `GET` as `LOC`, a date string as `LOC`. See
[Why the baseline fails](#why-the-baseline-fails) for the full breakdown.

**The 1k run learned format before it learned the task.** Strict JSON parsing
jumped from 0% to 92.5%, but recall actually *declined* (0.195 → 0.175) and
out-of-schema labels *increased* (11 → 15). Inspecting the failures showed the
model half-acquiring the output shape — emitting `[ROME]` or `[JSON]`, correct
vocabulary inside incorrect structure. At 900 training examples over 3 epochs,
the adapter had learned what the answer looks like but not how to produce it.

**Error analysis pointed to recall, not label confusion.** Of 406 gold entities
in the 1k run: 71 exact matches, 51 found with the wrong type, and 284 not found
at all. Median output length was 36 characters against a 256-token cap, so
truncation was ruled out — the model was stopping early rather than running long.
Type distribution showed it had effectively dropped a class, predicting 1 `MISC`
against 34 in the gold labels, and under-predicting `LOC` 25 to 130.

**More data fixed it.** Scaling to 8k examples moved recall from 0.175 to 0.837
and eliminated out-of-schema labels entirely. The invented lowercase types the
1k model produced (`person`, `country`, `city`, `year`, `verb`) disappeared
completely.

## Why the baseline fails

The aggregate metrics (0.141 F1, 0% strict JSON) undersell how specific the
failure modes are. Five distinct, compounding problems are visible in the raw
generations (`results/baseline.json`):

- **Format.** 0/200 outputs are a bare JSON array as instructed. 159/200 wrap
  the JSON in markdown code fences; the remaining 41/200 collapse
  single-entity sentences to a bare object (`{"text": ..., "type": ...}`)
  instead of a one-element list.
- **Over-generation.** The model predicts 714 spans against 406 gold
  entities — 76% more — skewing toward whole clauses rather than entities
  (avg. predicted span 1.78 words, max 12, vs. gold's avg. 1.46 words, max
  3). It tags `"China controlled most of the match"` as `PER`, for example.
  90.6% of predicted span text is a verbatim substring of the input, so this
  is over-broad extraction, not hallucinated text.
- **Type default bias.** `PER` accounts for 61.6% of predictions vs. 47.5% of
  gold labels — ambiguous spans default to `PER`. Of predictions that land on
  a real gold entity's exact text, 18.6% still have the wrong type.
- **Surface heuristic over-tagging.** 73.4% of non-sentence-initial
  capitalized words end up inside some predicted span, consistent with
  "capitalized = entity" rather than real boundary detection.
- **Schema collapse on out-of-distribution input.** All 11 out-of-schema type
  labels (`person`, `HRIO`, `motion`, `number`, `organization`) appear on
  CoNLL's terse cricket box-score sentences (e.g. `"C. Spearman c Moin Khan b
  Wasim 0"`), where the model abandons `PER`/`ORG`/`LOC`/`MISC` entirely and
  invents its own labels.

Net effect: this isn't "the model is bad at NER" so much as "the model was
never taught this task or output contract." It defaults to generic
instruction-tuned chat behavior — markdown formatting, clause-level salience,
capitalization heuristics — that resembles NER without being it.

## Known limitation: label noise in the test set

CoNLL-2003 contains documented annotation errors. Wang et al. (2019) manually
corrected the full test set and found 186 sentences — 5.38% of the data —
containing at least one token label error, releasing the corrected version as
CoNLL++.

This is visible in the first test sentence, which appears in the evaluation set:

```
SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .
gold: JAPAN=LOC, CHINA=PER
```

`CHINA` is labeled `PER` in the original gold annotations. Some fraction of the
remaining F1 gap is therefore unwinnable against this test set. Re-scoring
against CoNLL++ would give a cleaner number.

A follow-up correction, CleanCoNLL ([Rücker & Akbik, EMNLP
2023](https://aclanthology.org/2023.emnlp-main.533/)), fixed 7.0% of all
labels and found that 47% of what looked like model errors against the
original annotations were in fact correct predictions penalized by bad gold
labels; SOTA on the cleaned set rises to 97.1 F1. The commonly cited ~93-94
ceiling on the original test set is therefore partly an artifact of
annotation noise, not a hard limit on task difficulty.

## Related work

**No prior public work fine-tunes an LFM model on CoNLL-2003.** A search
across HuggingFace, GitHub, and Liquid AI's own blog, docs, and cookbook
turned up no model card, paper, or repo combining any LFM/LFM2/LFM2.5 model
with CoNLL-2003 or CoNLL++. The closest adjacent work converts LFM2.5 into a
*bidirectional encoder* for GLiNER-style span matching —
[SauerkrautLM-LFM2.5-GLiNER](https://huggingface.co/VAGOsolutions/SauerkrautLM-LFM2.5-GLiNER)
and Liquid's own
[LFM2.5-Encoder-350M-PII-Detector](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M-PII-Detector) —
evaluated on different benchmarks (CrossNER, PII datasets, BioNLP-CG), not
CoNLL. As far as public evidence shows, decoder-only causal SFT of LFM2.5 on
CoNLL-2003 via `leap-finetune` is a novel combination. (Caveat: this can't
rule out unpublished internal work.)

**Liquid AI already targets this problem space commercially, but not through
this recipe.** Liquid ships a dedicated "Extract" product line —
[LFM2-350M/1.2B-Extract](https://huggingface.co/LiquidAI/LFM2-1.2B-Extract)
and
[LFM2.5-VL-Extract](https://huggingface.co/LiquidAI/LFM2.5-VL-450M-Extract) —
for schema-driven JSON/XML/YAML extraction, and their [technical
report](https://arxiv.org/abs/2511.23404) names data extraction as a
first-class use case. But their own eval is an internal document-extraction
benchmark, not a public NER benchmark, and `leap-finetune` — the tool used
for this project — ships no NER or structured-extraction example config; its
reference tasks are generic SFT, GRPO math, VLM benchmarks, and document
classification. This project fills that gap rather than following an
official recipe.

**Tested directly: `LFM2-350M-Extract` doesn't solve this task zero-shot.**
Running the same 200 test sentences through `LiquidAI/LFM2-350M-Extract`
gives 0.069 F1 with a plain-language system prompt, and 0.055 F1 with a
schema-formatted prompt following Liquid's documented convention
(`extract_schema_prompt.txt`) — both well below the *untuned* base
`LFM2.5-350M`'s 0.141 F1, let alone the 0.871 this project's fine-tuned
adapter reaches. The schema-formatted prompt cut out-of-schema type
hallucinations from 100/200 to 31/200 but didn't improve F1, so this isn't
primarily a prompting problem. The failure pattern — whole clauses tagged as
a single entity, e.g. `"China controlled most of the match"` → `PER` —
suggests `Extract` is tuned for bounded document-field extraction (pull
`name`/`email`/`invoice_number`, one value per field) rather than open-set
span tagging, where an unknown number of entities, including repeats of the
same type, can appear anywhere in a sentence. Full outputs in
`results/extract-zeroshot.json` and `results/extract-zeroshot-schema.json`.

## Setup

Model: `LiquidAI/LFM2.5-350M` (post-trained, not the base checkpoint)
Method: LoRA (`DEFAULT_LORA`), SFT, 3 epochs, lr 2e-5
Training: leap-finetune on Modal, single A10G
Data: CoNLL-2003, converted to chat-format SFT with a fixed system prompt
Evaluation: 200 held-out test sentences, greedy decoding

### Reproducing

```bash
# 1. Convert CoNLL-2003 to LFM2 SFT format
uv run --with datasets python convert_conll.py

# 2. Push to HuggingFace Hub
#    (required — the Modal backend does not ship local files into the container)
uv run --with datasets --with huggingface_hub python -c "
from datasets import load_dataset
ds = load_dataset('json', data_files='data/train.jsonl', split='train')
ds.push_to_hub('<user>/conll2003-ner-sft-8k', private=True)
"

# 3. Train
cd ../leap-finetune && uv run leap-finetune ../lfm2-ner-poc/config.yaml

# 4. Download the merged checkpoint from the Modal volume
uv run modal volume get leap-finetune "<run_dir>" ../lfm2-ner-poc/model/

# 5. Evaluate
uv run --with torch --with transformers --with accelerate python eval_ner.py \
  --model "<merged_model_path>" --out results/tuned-8k.json
```

Full per-example model outputs for all three runs are in `results/`, including
raw generations, parsed predictions, and gold labels.

## Notes on the data

CoNLL-2003 is derived from Reuters newswire and is not redistributed here. The
converter pulls it from the HuggingFace Hub. Note that `eriktks/conll2003` ships
a Python loader script, which `datasets` v4 no longer supports; the converter
loads from the auto-generated parquet branch instead
(`revision="refs/convert/parquet"`).

## Output length

| | Baseline | 1k | 8k |
|---|---|---|---|
| Median response | 206 chars | 36 chars | 74 chars |
| Mean response | 210 chars | 42 chars | 75 chars |

The tuned model produces 64% shorter output than the baseline while scoring 6x
higher on F1. The baseline is verbose because it wraps everything in markdown
fences and tags nearly every capitalized token; the 8k model emits a bare array
containing only the entities it found. Fewer generated tokens means
proportionally lower inference cost, independent of hardware.

The 1k model is shortest of all, but for the wrong reason — it was
under-emitting entities, which is the same undertraining signal visible in its
recall and type distribution.

## What is not measured

Latency numbers were collected but are omitted from the headline results.
All inference ran at full precision on Apple Silicon MPS through
`transformers`, which is close to the slowest way to run this model — Liquid
reports 313 tok/s on AMD CPU with optimized runtimes. Re-running the baseline
eval twice on the identical 200 sentences gave median latencies of 3941.9ms
and 4254.2ms — a 7.9% swing from run-to-run noise alone, on the same machine,
same model, same inputs. A meaningful latency claim requires the GGUF or MLX
build.
