import json
from pathlib import Path
from datasets import load_dataset

SYSTEM = (
    "Extract named entities from the sentence. Return a JSON array where each "
    'element is an object with keys "text" and "type". "type" must be one of: '
    "PER, ORG, LOC, MISC. Return [] if there are no entities. "
    "Output only the JSON array, nothing else."
)

N_TRAIN = 1000
N_TEST = 200


def extract_spans(tokens, tags, names):
    out, cur, cur_type = [], [], None
    for tok, tag in zip(tokens, tags):
        label = names[tag]
        if label == "O":
            if cur:
                out.append({"text": " ".join(cur), "type": cur_type})
                cur, cur_type = [], None
            continue
        prefix, ent = label.split("-", 1)
        if prefix == "B" or cur_type != ent:
            if cur:
                out.append({"text": " ".join(cur), "type": cur_type})
            cur, cur_type = [tok], ent
        else:
            cur.append(tok)
    if cur:
        out.append({"text": " ".join(cur), "type": cur_type})
    return out


def build(split, names, n):
    rows = []
    for ex in split:
        tokens = ex["tokens"]
        if not tokens or tokens[0] == "-DOCSTART-":
            continue
        sentence = " ".join(tokens)
        gold = extract_spans(tokens, ex["ner_tags"], names)
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": sentence},
                    {"role": "assistant", "content": json.dumps(gold)},
                ]
            }
        )
        if len(rows) >= n:
            break
    return rows


def write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{path}: {len(rows)} rows")


CONLL_NAMES = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG",
               "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

ds = load_dataset("eriktks/conll2003", revision="refs/convert/parquet")
feat = ds["train"].features["ner_tags"]
names = getattr(getattr(feat, "feature", None), "names", None) or CONLL_NAMES
print("label names:", names)

Path("data").mkdir(exist_ok=True)
write("data/train.jsonl", build(ds["train"], names, N_TRAIN))
write("data/test.jsonl", build(ds["test"], names, N_TEST))
