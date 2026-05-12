import csv
from collections import Counter

rows = list(csv.DictReader(open("probe_summary.csv")))
n = len(rows)
G = int(rows[0]["n_rollouts"])
dist = Counter(int(r["n_correct"]) for r in rows)

print(f"Total questions: {n}")
print(f"G = {G} rollouts/question\n")

labels = {0: "unsolvable for this model", G: "trivial - model always gets it"}
print(f"{'n_correct':<11}{'count':>7}{'pct':>9}  meaning")
for k in range(G + 1):
    c = dist.get(k, 0)
    print(f"  {k}/{G}      {c:>7}{100*c/n:>8.2f}%  {labels.get(k, 'useful (variance)')}")

no_boxed = sum(1 for r in rows if r["any_boxed"].lower() == "false")
solved = sum(1 for r in rows if int(r["n_correct"]) > 0)
all_correct = dist.get(G, 0)
all_wrong = dist.get(0, 0)
mixed = n - all_correct - all_wrong

print()
print(f"At least one correct rollout: {solved}  ({100*solved/n:.2f}%)")
print(f"Zero correct rollouts (hard): {all_wrong}  ({100*all_wrong/n:.2f}%)")
print(f"All {G} correct (trivial):       {all_correct}  ({100*all_correct/n:.2f}%)")
print(f"Mixed 1-{G-1}/{G} (useful RL):       {mixed}  ({100*mixed/n:.2f}%)")
print(f"No boxed answer in any rollout: {no_boxed}  ({100*no_boxed/n:.2f}%)")
