import argparse, json, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", default="LiquidAI/LFM2.5-350M")
p.add_argument("--test", default="data/test.jsonl")
p.add_argument("--out", default="results/baseline.json")
p.add_argument("--limit", type=int, default=200)
a = p.parse_args()

dev = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev).eval()

rows = [json.loads(l) for l in open(a.test)][: a.limit]

def norm(items):
    return {(d["text"].strip(), d["type"].strip()) for d in items
            if isinstance(d, dict) and "text" in d and "type" in d}

valid = 0
tp = fp = fn = 0
lat = []
preds = []

for i, r in enumerate(rows):
    msgs = r["messages"][:2]
    gold = norm(json.loads(r["messages"][2]["content"]))
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(dev)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=256, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    lat.append((time.perf_counter() - t0) * 1000)
    text = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()
    try:
        parsed = json.loads(text)
        assert isinstance(parsed, list)
        pred = norm(parsed)
        valid += 1
    except Exception:
        pred = set()
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
       "schema_valid_pct": round(100 * valid / len(rows), 2),
       "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
       "median_latency_ms": round(lat[len(lat) // 2], 1), "device": dev}
print(json.dumps(res, indent=2))

import os
os.makedirs("results", exist_ok=True)
json.dump({"summary": res, "predictions": preds}, open(a.out, "w"), indent=2)
print(f"wrote {a.out}")
