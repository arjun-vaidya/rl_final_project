import csv
import json
import random
from collections import Counter

TARGET_TOTAL = 1000
TRIVIAL_SAMPLE_SEED = 0

rows = list(csv.DictReader(open("../data_probing/probe_summary.csv")))
G = int(rows[0]["n_rollouts"])

mixed, hard, trivial_pool = [], [], []
for r in rows:
    nc = int(r["n_correct"])
    idx = int(r["idx"])
    if nc == 0:
        hard.append(idx)
    elif nc < G:
        mixed.append(idx)
    else:
        trivial_pool.append(idx)

mixed.sort()
hard.sort()

n_trivial = max(0, TARGET_TOTAL - len(mixed) - len(hard))
rng = random.Random(TRIVIAL_SAMPLE_SEED)
trivial = sorted(rng.sample(trivial_pool, k=min(n_trivial, len(trivial_pool))))

partition = {
    "probed_with": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "dataset": "gsm8k/main/train",
    "probe": {"G": G, "temperature": 0.8},
    "target_total": TARGET_TOTAL,
    "trivial_sample_seed": TRIVIAL_SAMPLE_SEED,
    "counts": {
        "mixed": len(mixed),
        "hard": len(hard),
        "trivial": len(trivial),
        "all": len(mixed) + len(hard) + len(trivial),
        "probed_total": len(rows),
    },
    "mixed": mixed,
    "hard": hard,
    "trivial": trivial,
}

with open("partition.json", "w") as f:
    json.dump(partition, f, indent=2)

dist = Counter(int(r["n_correct"]) for r in rows)
print(f"Probed {len(rows)} GSM8K-train questions with G={G}")
print(f"  mixed (1-{G-1}/{G} correct): {len(mixed)}")
print(f"  hard  (0/{G} correct):       {len(hard)}")
print(f"  trivial pool ({G}/{G}):      {dist.get(G, 0)}  -> sampled {len(trivial)} (seed={TRIVIAL_SAMPLE_SEED})")
print(f"Wrote partition.json: {partition['counts']['all']} indices total")
