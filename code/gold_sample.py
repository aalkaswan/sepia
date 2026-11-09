"""Draw the gold-standard sample for measuring the prompt detector's recall.

The sample is stratified by submission year and by whether the detector found
prompts in the paper, with proportional allocation, so it can be analysed without
weights. Papers reviewed while building the detector (the dev rounds in
data/prompts/iterations/sampled_ids.txt) are excluded.

Annotators must be blind to the detector, so the strata (which reveal it) go to a
separate file:
    data/gold/sample.csv   paper_id, title, year, pdf_url      (for annotators)
    data/gold/strata.csv   paper_id, year, found               (for the analysis)

Usage: python3 gold_sample.py [--margin 0.10]
"""
import argparse
import csv
import glob
import json
import os
import random

from paper_numbers import GOLD_MARGIN, sample_size

SEED = 2026
OUT_DIR = "data/gold"


def allocate(sizes, n):
    """Split n across strata in proportion to their sizes (largest remainder)."""
    total = sum(sizes.values())
    exact = {k: n * v / total for k, v in sizes.items()}
    alloc = {k: int(x) for k, x in exact.items()}
    for k in sorted(exact, key=lambda k: exact[k] - alloc[k], reverse=True)[:n - sum(alloc.values())]:
        alloc[k] += 1
    return alloc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--margin", type=float, default=GOLD_MARGIN)
    args = parser.parse_args()

    papers = {r["arxiv_id"]: r for r in csv.DictReader(open("arxiv_papers.csv", encoding="utf-8"))}
    missing = {r["arxiv_id"] for r in csv.DictReader(open("data/prompts/missing_from_snapshot.csv", encoding="utf-8"))}
    searched = [pid for pid in papers if pid not in missing]
    dev = set(open("data/prompts/iterations/sampled_ids.txt").read().split())
    latest = sorted(glob.glob("data/prompts/iterations/iter_*/hits.jsonl"))[-1]
    found = {json.loads(line)["id"] for line in open(latest)}

    n = sample_size(len(searched), args.margin)
    strata = {}
    for pid in searched:
        if pid not in dev:
            strata.setdefault((papers[pid]["submitted"][:4], pid in found), []).append(pid)
    alloc = allocate({k: len(v) for k, v in strata.items()}, n)

    # Each stratum is shuffled once and the sample takes its first papers, so a
    # larger sample (a smaller margin) always contains a smaller one.
    rng = random.Random(SEED)
    sample = []
    for key in sorted(strata):
        order = sorted(strata[key])
        rng.shuffle(order)
        sample += [(pid, *key) for pid in order[:alloc[key]]]
    random.Random(SEED).shuffle(sample)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "sample.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["paper_id", "title", "year", "pdf_url"])
        for pid, year, _ in sample:
            w.writerow([pid, papers[pid]["title"], year, f"https://arxiv.org/pdf/{pid}"])
    with open(os.path.join(OUT_DIR, "strata.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["paper_id", "year", "found"])
        for pid, year, is_found in sample:
            w.writerow([pid, year, int(is_found)])

    print(f"population {len(searched)} (dev excluded: {len(searched) - sum(map(len, strata.values()))}), "
          f"margin {args.margin} -> n = {n}, detector run {latest}")
    for key in sorted(strata):
        print(f"  {key[0]} {'found    ' if key[1] else 'not found'}  {len(strata[key]):6d} papers -> {alloc[key]}")


if __name__ == "__main__":
    main()
