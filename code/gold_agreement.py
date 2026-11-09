"""Agreement between the gold-standard annotators, and the detector scored against the gold standard.

Reads the two annotators' labels exported from the annotation page's database
(data/gold/annotations/annotations/<annotator>/papers/<paper_id>.json), the sample
and its strata (written by gold_sample.py), and the detector's latest hits.

Paper level: does the paper show at least one prompt? Agreement is Cohen's kappa
and Krippendorff's alpha (nominal). Disagreements go to data/gold/disagreements.csv;
once they are resolved, data/gold/adjudicated.csv (paper_id, has_prompts, note)
overrides them and the detector is scored against the resolved labels.

Prompt level: once the annotators' lists are reconciled (gold_reconcile.py),
a gold prompt counts as found when its first words appear in one of the
detector's blocks for that paper (compared as lowercase letters only, since
text copied from a PDF loses spaces and picks up line numbers, and LaTeX adds
markup). The interval is a bootstrap over papers; the annotators' two lists
also give a capture-recapture estimate of how many prompts both missed.

Usage: python3 gold_agreement.py
"""
import csv
import glob
import json
import math
import os
import random
import re
from difflib import SequenceMatcher

import paper_numbers

GOLD_DIR = "data/gold"
ANNOTATIONS = os.path.join(GOLD_DIR, "annotations", "annotations")
# Characters of an annotated prompt's opening that must appear in a detector block.
MATCH_CHARS = 40
# The reconciled gold prompts, and the bootstrap resamples for the prompt-level recall interval.
PROMPTS = os.path.join(GOLD_DIR, "prompts.csv")
BOOTSTRAP = 10000
COMMAND_NAME = re.compile(r"\\[A-Za-z]+\*?")
NOT_ALNUM = re.compile(r"[^a-z0-9]")
DIGITS = re.compile(r"[0-9]")


def norm(text):
    return NOT_ALNUM.sub("", COMMAND_NAME.sub("", text).lower())


def load_annotators():
    """{annotator: {paper_id: annotation}} for papers in the sample that are marked done."""
    sample = {r["paper_id"] for r in csv.DictReader(open(os.path.join(GOLD_DIR, "sample.csv"), encoding="utf-8"))}
    out = {}
    for path in sorted(glob.glob(os.path.join(ANNOTATIONS, "*", "papers"))):
        docs = (json.load(open(f)) for f in glob.glob(os.path.join(path, "*.json")))
        out[path.split(os.sep)[-2]] = {d["paper_id"]: d for d in docs if d["paper_id"] in sample and d.get("done")}
    return sample, out


def cohen_kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in set(a) | set(b))
    return (po - pe) / (1 - pe)


def krippendorff_alpha(a, b):
    """Nominal alpha for two coders with no missing values."""
    values = a + b
    n = len(values)
    disagree_obs = 2 * sum(x != y for x, y in zip(a, b))
    disagree_exp = sum(values.count(c) * values.count(k) for c in set(values) for k in set(values) if c != k)
    return 1 - (n - 1) * disagree_obs / disagree_exp


def matches(prompt, blocks):
    """Whether the opening of an annotated prompt appears in one of the blocks.

    Digits are ignored, since first words copied from a PDF pick up listing line
    numbers, and the second MATCH_CHARS characters are tried too, since the
    copied opening can start with a label such as "[Initial prompt:]".
    """
    text = DIGITS.sub("", norm(prompt))
    blocks = [DIGITS.sub("", b) for b in blocks]
    for want in (text[:MATCH_CHARS], text[MATCH_CHARS:2 * MATCH_CHARS]):
        if len(want) < MATCH_CHARS and want != text:
            continue
        for block in blocks:
            m = SequenceMatcher(None, want, block, autojunk=False).find_longest_match(0, len(want), 0, len(block))
            if want and m.size >= len(want) * 0.8:
                return True
    return False


def chapman(n_a, n_b, both, seen):
    """Capture-recapture with the two annotators as the two captures: Chapman's estimate of the
    number of prompts, with Chao's log-transformed interval, which never falls below the `seen` prompts."""
    est = (n_a + 1) * (n_b + 1) / (both + 1) - 1
    unseen = est - seen
    if unseen <= 0:
        return est, (seen, seen)
    var = (n_a + 1) * (n_b + 1) * (n_a - both) * (n_b - both) / ((both + 1) ** 2 * (both + 2))
    c = math.exp(paper_numbers.Z * math.sqrt(math.log(1 + var / unseen ** 2)))
    return est, (seen + unseen / c, seen + unseen * c)


def prompt_recall(blocks):
    """The detector against the reconciled gold prompts (data/gold/prompts.csv, written by gold_reconcile.py)."""
    rows = list(csv.DictReader(open(PROMPTS, encoding="utf-8")))
    per_paper = {}
    for r in rows:
        k, n = per_paper.get(r["paper_id"], (0, 0))
        per_paper[r["paper_id"]] = (k + matches(r["first_words"], blocks.get(r["paper_id"], [])), n + 1)
    found = sum(k for k, _ in per_paper.values())
    # Prompts in one paper are not independent, so the interval resamples papers.
    rng, papers, reps = random.Random(0), list(per_paper.values()), []
    for _ in range(BOOTSTRAP):
        draw = [rng.choice(papers) for _ in papers]
        reps.append(sum(k for k, _ in draw) / sum(n for _, n in draw))
    reps.sort()
    n_a = sum(r["by_a"] == "1" for r in rows)
    n_b = sum(r["by_b"] == "1" for r in rows)
    both = sum(r["by_a"] == "1" and r["by_b"] == "1" for r in rows)
    est, (lo, hi) = chapman(n_a, n_b, both, len(rows))
    # The same estimate summed over where the prompts are, a check against some places being harder to spot.
    by_where = {}
    for r in rows:
        by_where.setdefault(r["where"], []).append(r)
    stratified = sum(chapman(sum(r["by_a"] == "1" for r in rr), sum(r["by_b"] == "1" for r in rr),
                             sum(r["by_a"] == "1" and r["by_b"] == "1" for r in rr), len(rr))[0]
                     for rr in by_where.values())
    return {
        "prompts": len(rows), "found": found, "recall": found / len(rows),
        "recall_ci": (reps[int(0.025 * BOOTSTRAP)], reps[int(0.975 * BOOTSTRAP) - 1]),
        "in_flagged": sum(n for p, (_, n) in per_paper.items() if p in blocks),
        "by_a": n_a, "by_b": n_b, "both": both,
        "estimated": est, "estimated_ci": (lo, hi), "estimated_by_where": stratified,
        "missed": [(r["paper_id"], r["entries"], r["first_words"][:80]) for r in rows
                   if not matches(r["first_words"], blocks.get(r["paper_id"], []))],
    }


def summary():
    sample, annotators = load_annotators()
    (ua, la), (ub, lb) = annotators.items()
    papers = sorted(sample & la.keys() & lb.keys())
    a = [la[p]["has_prompts"] for p in papers]
    b = [lb[p]["has_prompts"] for p in papers]
    disagree = [p for p, x, y in zip(papers, a, b) if x != y]

    adjudicated = {}
    adj_path = os.path.join(GOLD_DIR, "adjudicated.csv")
    if os.path.exists(adj_path):
        adjudicated = {r["paper_id"]: r["has_prompts"].strip().lower() in ("1", "true", "yes")
                       for r in csv.DictReader(open(adj_path, encoding="utf-8")) if r["has_prompts"].strip()}
    unresolved = [p for p in disagree if p not in adjudicated]
    gold = {p: adjudicated.get(p, la[p]["has_prompts"]) for p in papers if p not in unresolved}

    found = {r["paper_id"]: r["found"] == "1" for r in csv.DictReader(open(os.path.join(GOLD_DIR, "strata.csv")))}
    tp = sum(1 for p, g in gold.items() if g and found[p])
    fp = sum(1 for p, g in gold.items() if not g and found[p])
    fn = sum(1 for p, g in gold.items() if g and not found[p])

    latest = sorted(glob.glob("data/prompts/iterations/iter_*/hits.jsonl"))[-1]
    blocks = {}
    for line in open(latest):
        d = json.loads(line)
        if d["id"] in sample:
            blocks[d["id"]] = [norm(h["text"]) for h in d["hits"]]
    prompt_level = prompt_recall(blocks) if os.path.exists(PROMPTS) else None
    # Missed papers with a prompt written as text in the LaTeX source (not only in an image).
    text_papers = ({r["paper_id"] for r in csv.DictReader(open(PROMPTS, encoding="utf-8")) if r["form"] == "text"}
                   if os.path.exists(PROMPTS) else set())

    return {
        "annotators": [ua, ub], "papers": len(papers),
        "positives": [sum(a), sum(b)], "agree": len(papers) - len(disagree),
        "kappa": cohen_kappa(a, b), "alpha": krippendorff_alpha(a, b),
        "disagreements": disagree, "unresolved": unresolved,
        "gold_papers": len(gold), "gold_positives": sum(gold.values()),
        "tp": tp, "fp": fp, "fn": fn,
        "unflagged": sum(1 for p in gold if not found[p]),
        "missed_text": sum(1 for p, g in gold.items() if g and not found[p] and p in text_papers),
        "precision": tp / (tp + fp) if tp + fp else None, "precision_ci": paper_numbers.wilson(tp, tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None, "recall_ci": paper_numbers.wilson(tp, tp + fn) if tp + fn else None,
        "prompt_level": prompt_level, "detector_run": latest,
    }


def main():
    s = summary()
    _, annotators = load_annotators()
    (ua, la), (ub, lb) = annotators.items()
    with open(os.path.join(GOLD_DIR, "disagreements.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["paper_id", f"has_prompts_{ua}", f"prompts_{ua}", f"notes_{ua}",
                    f"has_prompts_{ub}", f"prompts_{ub}", f"notes_{ub}"])
        for p in s["disagreements"]:
            w.writerow([p, int(la[p]["has_prompts"]), len(la[p]["prompts"]), la[p]["notes"],
                        int(lb[p]["has_prompts"]), len(lb[p]["prompts"]), lb[p]["notes"]])
    # A template for the resolved labels; existing resolutions are kept.
    adj_path = os.path.join(GOLD_DIR, "adjudicated.csv")
    if not os.path.exists(adj_path):
        with open(adj_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["paper_id", "has_prompts", "note"])
            for p in s["disagreements"]:
                w.writerow([p, "", ""])
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
