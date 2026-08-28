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

Published state of the art on CoNLL-2003 is roughly 0.93 F1, so 0.871 from a
350M model with a LoRA adapter is near the practical ceiling for this setup.

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
`SOCCER` as `PER`, `GET` as `LOC`, a date string as `LOC`.

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

## What is not measured

Latency numbers were collected but are omitted. All inference ran at full
precision on Apple Silicon MPS through `transformers`, which is close to the
slowest way to run this model — Liquid reports 313 tok/s on AMD CPU with
optimized runtimes. Measured latency varied ~8% between identical runs on the
same machine. A meaningful latency claim requires the GGUF or MLX build.
