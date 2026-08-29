"""Re-score saved predictions against CoNLL++ and report bootstrap CIs.

CoNLL-2003's test set contains documented annotation errors. Wang et al. (2019)
released a manually corrected version, CoNLL++. This re-scores the predictions
already saved in results/ against both label sets, so no model inference is
needed.

The bootstrap resamples sentences, not entities: entities within a sentence are
correlated, so resampling entities would understate the interval. Note that the
interval covers test-set sampling only. It says nothing about training or seed
variance, since each run here is a single training run at one seed.

Usage: uv run --with nothing python rescore_conllpp.py   (needs network)
"""
import json
import random
import urllib.request

CONLLPP_URL = (
    "https://raw.githubusercontent.com/ZihanWangKi/CrossWeigh/"
    "master/data/conllpp_test.txt"
)
RUNS = ["results/baseline.json", "results/tuned-1k.json", "results/tuned-8k.json"]
TEST = "data/test.jsonl"
N_BOOT = 10000
SEED = 0


def extract_spans(tokens, labels):
    """Same span logic as convert_conll.py, over raw CoNLL columns."""
    out, cur, cur_type = [], [], None
    for tok, label in zip(tokens, labels):
        if label == "O":
            if cur:
                out.append((" ".join(cur), cur_type))
                cur, cur_type = [], None
            continue
        prefix, ent = label.split("-", 1)
        if prefix == "B" or cur_type != ent:
            if cur:
                out.append((" ".join(cur), cur_type))
            cur, cur_type = [tok], ent
        else:
            cur.append(tok)
    if cur:
        out.append((" ".join(cur), cur_type))
    return out


def load_conllpp():
    raw = urllib.request.urlopen(CONLLPP_URL).read().decode("utf-8")
    sents, toks, labs = [], [], []
    for line in raw.split("\n"):
        if not line.strip():
            if toks:
                sents.append((toks, labs))
                toks, labs = [], []
            continue
        parts = line.split()
        if parts[0] == "-DOCSTART-":
            continue
        toks.append(parts[0])
        labs.append(parts[-1])
    if toks:
        sents.append((toks, labs))
    return [s for s in sents if s[0]]


def prf(triples):
    tp = sum(t[0] for t in triples)
    fp = sum(t[1] for t in triples)
    fn = sum(t[2] for t in triples)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def boot_ci(triples, n=N_BOOT, seed=SEED):
    rng = random.Random(seed)
    N = len(triples)
    vals = sorted(prf([triples[rng.randrange(N)] for _ in range(N)])[2]
                  for _ in range(n))
    return vals[int(0.025 * n)], vals[int(0.975 * n)]


rows = [json.loads(line) for line in open(TEST)]
pp = load_conllpp()

# Match positionally, not by text: the test set contains duplicate sentence
# strings (e.g. the dateline "LONDON 1996-12-06" appears five times), so a
# text-keyed lookup would silently attach the wrong labels.
assert len(pp) >= len(rows), "CoNLL++ file shorter than the test slice"
for i, r in enumerate(rows):
    assert " ".join(pp[i][0]) == r["messages"][1]["content"], (
        f"alignment broke at index {i}; test.jsonl must be the first "
        f"{len(rows)} non-DOCSTART sentences of the CoNLL-2003 test split"
    )

pp_gold = [set(extract_spans(*pp[i])) for i in range(len(rows))]
orig_gold = [{(d["text"], d["type"])
              for d in json.loads(r["messages"][2]["content"])} for r in rows]

changed = [i for i in range(len(rows)) if orig_gold[i] != pp_gold[i]]
print(f"sentences whose gold labels change under CoNLL++: "
      f"{len(changed)}/{len(rows)}")
for i in changed:
    print(f"  [{i}] {rows[i]['messages'][1]['content'][:60]}")
    print(f"       was {sorted(orig_gold[i] - pp_gold[i])} "
          f"now {sorted(pp_gold[i] - orig_gold[i])}")
print()

for path in RUNS:
    d = json.load(open(path))
    preds = [{tuple(x) for x in p["pred"]} for p in d["predictions"]]
    print(f"=== {path} ===")
    for tag, golds in (("CoNLL-2003", orig_gold), ("CoNLL++", pp_gold)):
        tr = [(len(p & g), len(p - g), len(g - p))
              for p, g in zip(preds, golds)]
        prec, rec, f1 = prf(tr)
        lo, hi = boot_ci(tr)
        print(f"  {tag:11s} P={prec:.4f} R={rec:.4f} F1={f1:.4f}"
              f"   95% CI [{lo:.4f}, {hi:.4f}]")
    print()
