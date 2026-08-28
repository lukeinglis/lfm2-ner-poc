import argparse, json, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", default="LiquidAI/LFM2.5-350M")
p.add_argument("--test", default="data/test.jsonl")
p.add_argument("--out", default="results/baseline.json")
p.add_argument("--limit", type=int, default=200)
p.add_argument("--system-file", default=None,
                help="Override the system prompt with this file's contents "
                     "(default: use each row's own system message)")
a = p.parse_args()

dev = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev).eval()

rows = [json.loads(l) for l in open(a.test)][: a.limit]
system_override = open(a.system_file).read().strip() if a.system_file else None

def norm(items):
    return {(d["text"].strip(), d["type"].strip()) for d in items
            if isinstance(d, dict) and "text" in d and "type" in d}

def parse(text, lenient):
    s = text.strip()
    if lenient and s.startswith("```"):
        s = s.split("```")[1]
        if s.startswith("json"):
            s = s[4:]
        s = s.strip()
    obj = json.loads(s)
    if isinstance(obj, dict):
        if not lenient:
            raise ValueError("not a list")
        list_valued = [v for v in obj.values() if isinstance(v, list)]
        obj = list_valued[0] if len(list_valued) == 1 else [obj]
    if not isinstance(obj, list):
        raise ValueError("not a list")
    return obj

VALID_TYPES = {"PER", "ORG", "LOC", "MISC"}
bad_type = 0
strict = 0
valid = 0
tp = fp = fn = 0
lat = []
preds = []

for i, r in enumerate(rows):
    msgs = r["messages"][:2]
    if system_override is not None:
        msgs = [{"role": "system", "content": system_override}, msgs[1]]
    gold = norm(json.loads(r["messages"][2]["content"]))
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    enc = {k: v.to(dev) for k, v in enc.items()}
    in_len = enc["input_ids"].shape[-1]
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=256, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    lat.append((time.perf_counter() - t0) * 1000)
    text = tok.decode(out[0][in_len:], skip_special_tokens=True).strip()
    try:
        parse(text, lenient=False)
        strict += 1
    except Exception:
        pass
    try:
        pred = norm(parse(text, lenient=True))
        valid += 1
    except Exception:
        pred = set()
    bad_type += sum(1 for _, t in pred if t not in VALID_TYPES)
    tp += len(pred & gold); fp += len(pred - gold); fn += len(gold - pred)
    preds.append({"input": msgs[1]["content"], "gold": sorted(gold),
                  "raw": text, "pred": sorted(pred)})
    if (i + 1) % 25 == 0:
        print(f"{i+1}/{len(rows)}")

prec = tp / (tp + fp) if tp + fp else 0.0
rec = tp / (tp + fn) if tp + fn else 0.0
f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
lat.sort()

res = {"model": a.model, "n": len(rows),
       "strict_json_pct": round(100 * strict / len(rows), 2),
       "lenient_json_pct": round(100 * valid / len(rows), 2),
       "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
       "out_of_schema_types": bad_type,
       "median_latency_ms": round(lat[len(lat) // 2], 1), "device": dev}
print(json.dumps(res, indent=2))

import os
os.makedirs("results", exist_ok=True)
json.dump({"summary": res, "predictions": preds}, open(a.out, "w"), indent=2)
print(f"wrote {a.out}")
