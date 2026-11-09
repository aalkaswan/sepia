"""Write paper/generated/numbers.tex from the pipeline's outputs.

Reads arxiv_papers.csv, the snapshot-miss list, the latest collect_prompts.py
iteration and the review labels saved under data/prompts/iterations/iter_NN/review_db.

Usage: python3 paper_numbers.py
"""
import csv
import datetime
import glob
import json
import math
import os
import statistics

import collect_prompts
import gold_agreement

ITER_DIR = "data/prompts/iterations"
OUT = "paper/generated/numbers.tex"
# Confidence level for every interval and sample size in the paper.
CONFIDENCE, Z = "95", 1.96
# Margin of error the gold-standard sample is sized for.
GOLD_MARGIN = 0.10
# Cluster validation. While clustering is iterated, one coder screens each changed
# cluster with a double sampling plan: SCREEN_FIRST random prompts, accepted if all fit
# and revised if two do not; with exactly one misfit, SCREEN_SECOND more, accepted if
# all of those fit. The coder stops as soon as the outcome is settled. The acceptance
# chances and expected labels are reported for a good and a bad cluster (share of
# misfits). Once clusters are frozen, both coders label a fresh sample sized for
# CLUSTER_MARGIN, the same margin as the gold standard.
SCREEN_FIRST, SCREEN_SECOND = 5, 5
SCREEN_GOOD, SCREEN_BAD = 0.05, 0.30
CLUSTER_MARGIN = GOLD_MARGIN
CLUSTER_DIR = "data/clusters"
# The validation labels the major groups of the task types; prompts of this catch-all
# group are reviewed with the unassigned ones, as fitting none of them.
REVIEW_NONE_GROUP = "Other"
# SE activities, task types and clusters whose trend the results discuss: \trend<key>From and \trend<key>To.
TREND_NAMES = {"Problem": "Programming problem solving", "Summary": "Code summarization and explanation",
               "Judge": "LLM-as-judge evaluation and scoring", "Trajectory": "Agent trajectory analysis and verification",
               "Agent": "LLM as agent", "Security": "Security and binary analysis", "Testing": "Testing",
               "Debugging": "Debugging, fault localization and repair",
               "Development": "Software development", "LLMCentric": "LLM-centric tasks"}
# SE activities whose share of papers the results report: \groupShare<key>.
GROUP_NAMES = {"Development": "Software development", "QA": "Software quality assurance",
               "Maintenance": "Software maintenance", "LLM": "LLM-centric tasks",
               "Requirements": "Requirements engineering", "Design": "Software design",
               "Management": "Software management"}
# Listed papers the LaTeX snapshot lacked; those fetched later (ADDED_PAPERS, with how
# their TeX was assembled) are searched with the frozen detector (ADDED_HITS).
SNAPSHOT_MISSING = "data/prompts/missing_from_snapshot.csv"
ADDED_PAPERS = "data/prompts/added/papers.csv"
ADDED_HITS = "data/prompts/added/hits.jsonl"
# Shown in the paper for numbers whose pipeline stage has not run yet.
PENDING = r"\textbf{??}"


def wilson(k, n, z=Z):
    """Wilson score interval for k successes out of n."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def sample_size(population, margin, z=Z, p=0.5):
    """Papers needed for a proportion within +-margin, with finite-population correction."""
    n0 = z * z * p * (1 - p) / (margin * margin)
    return math.ceil(n0 / (1 + (n0 - 1) / population))


def binom(n, k, p):
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


def accept_chance(misfit_rate):
    """Chance that the screening plan accepts a cluster with this share of misfits."""
    p = misfit_rate
    return binom(SCREEN_FIRST, 0, p) + binom(SCREEN_FIRST, 1, p) * (1 - p) ** SCREEN_SECOND


def screen_expected_labels(misfit_rate):
    """Expected prompts labelled per cluster, stopping once the outcome is settled."""
    p = misfit_rate
    # The t-th prompt of the first stage is labelled while fewer than two misfits came before it;
    # the t-th of the second, after exactly one misfit in the first stage and none since.
    first = sum(binom(t - 1, 0, p) + binom(t - 1, 1, p) for t in range(1, SCREEN_FIRST + 1))
    second = binom(SCREEN_FIRST, 1, p) * sum((1 - p) ** (t - 1) for t in range(1, SCREEN_SECOND + 1))
    return first + second


def cluster_numbers(prompts, pct):
    """Macros for the clustering stage.

    Reads data/clusters/summary.json ({"prompts", "clusters", "unassigned", "hdbscan",
    "hdbscan_grid"}: the prompts clustered after block review and deduplication, the
    final number of clusters, the prompts HDBSCAN left unassigned, and the HDBSCAN
    setting chosen by DBCV among those tried) and data/clusters/validation.csv
    (prompt_id, cluster, fit_a, fit_b, fit: each coder's judgement of whether the
    prompt fits its cluster, 1/0, or x for a block that is not a prompt, and the
    resolved one; for an unassigned prompt, cluster -1, 1 means it fits none). Numbers whose file does not exist
    yet are shown as PENDING; the validation sample is sized on the extracted blocks
    until the summary exists.
    """
    summary_path = os.path.join(CLUSTER_DIR, "summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else None
    n = summary["prompts"] if summary else prompts
    assigned_path = os.path.join(CLUSTER_DIR, "assigned.json")
    dataset_prompts = json.load(open(assigned_path))["prompts"] if os.path.exists(assigned_path) else None
    values = {
        "clusterScreenFirst": str(SCREEN_FIRST),
        "clusterScreenSecond": str(SCREEN_SECOND),
        "clusterScreenGoodRate": pct(SCREEN_GOOD),
        "clusterScreenBadRate": pct(SCREEN_BAD),
        "clusterScreenAcceptGood": pct(accept_chance(SCREEN_GOOD)),
        "clusterScreenRejectBad": pct(1 - accept_chance(SCREEN_BAD)),
        "clusterScreenExpectedGood": f"{screen_expected_labels(SCREEN_GOOD):.1f}",
        "clusterMargin": f"{100 * CLUSTER_MARGIN:.0f}\\%",
        # The validation sample is drawn from every prompt, the added papers' included.
        "clusterSample": str(sample_size(dataset_prompts or n, CLUSTER_MARGIN)),
        "numClustered": f"{n:,}".replace(",", "{,}") if summary else PENDING,
        "numClusters": str(summary["clusters"]) if summary else PENDING,
        "clusterUnassigned": pct(summary["unassigned"] / summary["prompts"]) if summary else PENDING,
    }
    # The HDBSCAN settings tried, the one chosen by DBCV, and the unassigned share across them.
    grid = summary["hdbscan_grid"] if summary else []
    shares = [g["unassigned"] / summary["prompts"] for g in grid]
    values.update({
        "clusterGridSettings": str(len(grid)) if grid else PENDING,
        "clusterMinSize": str(summary["hdbscan"]["min_cluster_size"]) if grid else PENDING,
        "clusterDBCV": f"{summary['hdbscan']['dbcv']:.2f}" if grid else PENDING,
        "clusterUnassignedRange": f"{pct(min(shares))}--{pct(max(shares))}" if grid else PENDING,
        "clusterInitial": str(summary["hdbscan"]["clusters"]) if grid else PENDING,
        "clusterDupThreshold": f"{summary['dup_threshold']:.2f}" if summary else PENDING,
    })
    # Manual edits made during screening (data/clusters/edits.json).
    edits_path = os.path.join(CLUSTER_DIR, "edits.json")
    ops = json.load(open(edits_path))["ops"] if os.path.exists(edits_path) else []
    for op, macro in (("merge", "clusterEditMerges"), ("split", "clusterEditSplits"), ("dissolve", "clusterEditDissolves")):
        values[macro] = str(sum(op in o for o in ops)) if ops else PENDING
    # Screening effort: rounds written and prompts labelled in them.
    screen_dir = os.path.join(CLUSTER_DIR, "screening")
    rounds = sorted(f for f in os.listdir(screen_dir) if f.endswith(".csv")) if os.path.isdir(screen_dir) else []
    labels = [r["fits"].strip() for f in rounds for r in csv.DictReader(open(os.path.join(screen_dir, f), encoding="utf-8"))
              if r["fits"].strip()]
    values.update({"clusterScreenRounds": str(len(rounds)) if rounds else PENDING,
                   "clusterScreenLabelled": str(len(labels)) if labels else PENDING,
                   "clusterScreenNotPrompts": str(labels.count("x")) if labels else PENDING,
                   "numNotPrompts": str(summary["not_prompts"]) if summary else PENDING,
                   # Clusters whose screening found mostly non-prompts, and the blocks they took along.
                   "numDroppedClusters": str(summary["dropped_clusters"]) if summary else PENDING,
                   "numDropped": f"{summary['dropped']:,}".replace(",", "{,}") if summary else PENDING})
    validation_path = os.path.join(CLUSTER_DIR, "validation.csv")
    rows = list(csv.DictReader(open(validation_path, encoding="utf-8"))) if os.path.exists(validation_path) else []
    # Scored only once both coders and the resolution have filled every row.
    if rows and all(r[k].strip() for r in rows for k in ("fit_a", "fit_b", "fit")):
        # Labels are 1 (fits), 0 (misfit) or x (not a prompt); accuracy is over the prompts.
        a = [r["fit_a"].strip() for r in rows]
        b = [r["fit_b"].strip() for r in rows]
        # Accuracy is over the prompts in a cluster; for an unassigned prompt, 1 means it
        # rightly fits none of the clusters, reported separately.
        judged = [r["fit"].strip() for r in rows if r["fit"].strip() != "x" and int(r["cluster"]) >= 0]
        none = [r["fit"].strip() for r in rows if r["fit"].strip() != "x" and int(r["cluster"]) < 0]
        lo, hi = wilson(judged.count("1"), len(judged))
        nlo, nhi = wilson(none.count("1"), len(none)) if none else (0, 0)
        # Precision of the released prompts: the share of the sample (drawn from them,
        # proportionally) that is a prompt.
        found = sum(r["fit"].strip() == "x" for r in rows)
        q_lo, q_hi = wilson(found, len(rows))
        precision = lambda q: 1 - q
        values.update({
            "clusterValidated": str(len(rows)),
            "clusterAccuracy": pct(judged.count("1") / len(judged)),
            "clusterAccuracyCI": f"{pct(lo)}--{pct(hi)}",
            "clusterValidatedAssigned": str(len(judged)),
            "clusterValidatedUnassigned": str(len(none)),
            "clusterUnassignedNone": pct(none.count("1") / len(none)) if none else "n/a",
            "clusterUnassignedNoneCI": f"{pct(nlo)}--{pct(nhi)}" if none else "n/a",
            "clusterAgree": pct(sum(x == y for x, y in zip(a, b)) / len(rows)),
            # kappa is undefined when both coders judge every prompt the same way
            "clusterKappa": f"{gold_agreement.cohen_kappa(a, b):.2f}" if len(set(a + b)) > 1 else "n/a",
            "blockPrecision": pct(precision(found / len(rows))),
            "blockPrecisionCI": f"{pct(precision(q_hi))}--{pct(precision(q_lo))}",
        })
    else:
        values.update({k: PENDING for k in ("clusterValidated", "clusterAccuracy", "clusterAccuracyCI", "clusterAgree",
                                             "clusterKappa", "blockPrecision", "blockPrecisionCI",
                                             "clusterValidatedAssigned", "clusterValidatedUnassigned",
                                             "clusterUnassignedNone", "clusterUnassignedNoneCI")})
    return values


def task_numbers(pct):
    """Macros for the task types, from data/clusters/results.json (cluster_results.py).

    The three types used by the most papers, and the types whose share of papers
    rose and fell the most between the first and the last submission year.
    """
    path = os.path.join(CLUSTER_DIR, "results.json")
    keys = ["numTaskTypes", "numTaskGroups", "numReviewGroups", "numPromptPapers", "numFamilies", "numReusedFamilies",
            "taskYearFirst", "taskYearLast",
            "taskRiseName", "taskRiseFrom", "taskRiseTo", "taskFallName", "taskFallFrom", "taskFallTo"]
    keys += [f"task{r}{k}" for r in ("Top", "Second", "Third") for k in ("Name", "Share")]
    if not os.path.exists(path):
        return {k: PENDING for k in keys}
    res = json.load(open(path))
    # Families over every prompt, the added papers' included once they are assigned.
    assigned = os.path.join(CLUSTER_DIR, "assigned.json")
    summary = json.load(open(assigned if os.path.exists(assigned) else os.path.join(CLUSTER_DIR, "summary.json")))
    types = [t for t in res["types"] if t["cluster"] >= 0]
    first, last = (str(y) for y in (res["years"][0], res["years"][-1]))
    change = sorted(types, key=lambda t: t["by_year"][last] - t["by_year"][first])
    num = lambda x: f"{x:,}".replace(",", "{,}")
    values = {"numTaskTypes": str(len(types)), "numTaskGroups": str(len({t["group"] for t in types})),
              "numReviewGroups": str(len({t["group"] for t in types} - {REVIEW_NONE_GROUP})),
              "numPromptPapers": num(res["papers"]),
              "numFamilies": num(summary["families"]), "numReusedFamilies": str(summary["reused_families"]),
              "taskYearFirst": first, "taskYearLast": last,
              "taskRiseName": change[-1]["name"], "taskRiseFrom": pct(change[-1]["by_year"][first]),
              "taskRiseTo": pct(change[-1]["by_year"][last]),
              "taskFallName": change[0]["name"], "taskFallFrom": pct(change[0]["by_year"][first]),
              "taskFallTo": pct(change[0]["by_year"][last])}
    for r, t in zip(("Top", "Second", "Third"), types):
        values[f"task{r}Name"], values[f"task{r}Share"] = t["name"], pct(t["paper_share"])
    values["taskYearFirstPapers"] = num(res["papers_per_year"][first])
    values["taskYearLastPapers"] = num(res["papers_per_year"][last])
    # Share of the first and last year's papers for the types and clusters the results discuss.
    by_name = ({g["name"]: g for g in res.get("groups", [])} | {t["name"]: t for t in types}
               | {c["name"]: c for c in res.get("clusters", [])})
    for key, name in TREND_NAMES.items():
        if name in by_name:
            values[f"trend{key}From"] = pct(by_name[name]["by_year"][first])
            values[f"trend{key}To"] = pct(by_name[name]["by_year"][last])
    groups = {g["name"]: g for g in res.get("groups", [])}
    for key, group in GROUP_NAMES.items():
        if group in groups:
            values[f"groupShare{key}"] = pct(groups[group]["paper_share"])
    if "typed_papers" in res:
        values["multiTypeShare"] = pct(res["multi_type_papers"] / res["typed_papers"])
    return values


def dataset_numbers(pct):
    """Macros describing the released prompts: time span, prompts per paper, length, licences.

    The searched papers are the listed ones with LaTeX (in the snapshot or fetched later);
    a prompt is a block in data/clusters/blocks.jsonl still marked as one after the block
    review, including those of the added papers.
    """
    path = os.path.join(CLUSTER_DIR, "blocks.jsonl")
    keys = ["searchFrom", "searchTo", "promptPaperShare", "promptPaperShareFirst", "promptPaperShareLast",
            "promptsPerPaperMedian", "promptsPerPaperMax", "promptWordsMedian", "promptWordsIQR",
            "promptPapersOpen", "promptsOpen"]
    if not os.path.exists(path):
        return {k: PENDING for k in keys}
    missing = still_missing()
    papers = {r["arxiv_id"]: r for r in csv.DictReader(open("arxiv_papers.csv", encoding="utf-8"))
              if r["arxiv_id"] not in missing}
    prompts = [b for line in open(path) if (b := json.loads(line))["prompt"]]
    per_paper = {}
    for b in prompts:
        per_paper[b["paper_id"]] = per_paper.get(b["paper_id"], 0) + 1
    year = lambda pid: papers[pid]["submitted"][:4]
    years = sorted({year(p) for p in papers})
    share_in = lambda y: sum(year(p) == y for p in per_paper) / sum(year(p) == y for p in papers)
    q1, median, q3 = statistics.quantiles([len(b["text"].split()) for b in prompts], n=4)
    # Creative Commons and public-domain licences allow redistributing the prompt text.
    is_open = lambda pid: "creativecommons.org" in papers[pid]["license"]
    month = lambda d: datetime.date.fromisoformat(d).strftime("%B %Y")
    submitted = sorted(p["submitted"] for p in papers.values())
    return {
        "searchFrom": month(submitted[0]), "searchTo": month(submitted[-1]),
        "promptPaperShare": pct(len(per_paper) / len(papers)),
        "promptPaperShareFirst": pct(share_in(years[0])), "promptPaperShareLast": pct(share_in(years[-1])),
        "promptsPerPaperMedian": f"{statistics.median(per_paper.values()):g}",
        "promptsPerPaperMax": str(max(per_paper.values())),
        "promptWordsMedian": f"{median:.0f}", "promptWordsIQR": f"{q1:.0f}--{q3:.0f}",
        "promptPapersOpen": pct(sum(map(is_open, per_paper)) / len(per_paper)),
        "promptsOpen": pct(sum(is_open(b["paper_id"]) for b in prompts) / len(prompts)),
    }


def count_blocks(hits):
    """Number of non-overlapping prompt blocks among one paper's hits."""
    blocks, end = 0, -1
    for h in sorted(hits, key=lambda h: h["start"]):
        if h["start"] >= end:
            blocks += 1
        end = max(end, h["end"])
    return blocks


def added_latex():
    """Papers missing from the snapshot whose LaTeX was fetched later (build_added_dataset.py)."""
    if not os.path.exists(ADDED_PAPERS):
        return set()
    return {r["paper"] for r in csv.DictReader(open(ADDED_PAPERS, encoding="utf-8"))
            if r["resolution"] in ("single", "resolved", "fallback")}


def still_missing():
    """Listed papers with no LaTeX: missing from the snapshot and not fetched later."""
    snapshot = {r["arxiv_id"] for r in csv.DictReader(open(SNAPSHOT_MISSING, encoding="utf-8"))}
    return snapshot - added_latex()


def assign_numbers(pct):
    """Macros for the prompts of the papers added after the clusters were frozen
    (data/clusters/assigned.json, written by task_clusters.py assign)."""
    keys = ["numAddedPrompts", "numDatasetPrompts", "datasetUnassigned", "assignK", "assignCheck", "assignStable",
            "datasetUnassignedFirst", "secondPassGroupsFound", "secondPassSplits", "secondPassGroups", "secondPassPrompts",
            "secondPassCandidates", "secondPassMinSize", "secondPassMinFamilies", "numClustersTotal"]
    path = os.path.join(CLUSTER_DIR, "assigned.json")
    if not os.path.exists(path):
        return {k: PENDING for k in keys}
    a = json.load(open(path))
    num = lambda x: f"{x:,}".replace(",", "{,}")
    # Once the second pass has run, the prompts it placed are no longer unassigned.
    sp_path = os.path.join(CLUSTER_DIR, "second_pass_summary.json")
    sp = json.load(open(sp_path)) if os.path.exists(sp_path) else None
    clusters = json.load(open(os.path.join(CLUSTER_DIR, "clusters.json")))
    values = {
        "datasetUnassignedFirst": pct(a["unassigned"] / a["prompts"]),
        # HDBSCAN's groups, and the candidates left after the splits and the size cut.
        "secondPassGroupsFound": str(sp["groups_found"]) if sp else PENDING,
        "secondPassCandidates": str(sp["groups"]) if sp else PENDING,
        "secondPassMinSize": str(sp["hdbscan"]["min_cluster_size"]) if sp else PENDING,
        "secondPassMinFamilies": str(sp.get("min_families", "")) or PENDING if sp else PENDING,
        # Main clusters plus the second pass's groups.
        "numClustersTotal": str(sum(c["cluster"] >= 0 for c in clusters)),
        "secondPassSplits": str(sp["splits"]) if sp else PENDING,
        "secondPassGroups": str(sp["kept"]) if sp else PENDING,
        "secondPassPrompts": num(sp["assigned"]) if sp else PENDING,
    }
    return values | {
        "numAddedPrompts": num(a["added"]["prompts"]), "numDatasetPrompts": num(a["prompts"]),
        "datasetUnassigned": pct((sp or a)["unassigned"] / a["prompts"]), "assignK": str(a["added"]["k"]),
        # Leave-one-family-out: how often the vote gives a clustered prompt its own cluster.
        "assignCheck": pct(a["vote_check"]["agree"] / a["vote_check"]["prompts"]),
        # Added prompts voted into the same cluster with k = 5 and with k = 20.
        "assignStable": pct(min(a["added"]["same_k5"], a["added"]["same_k20"])),
    }


def main():
    listed = sum(1 for _ in csv.DictReader(open("arxiv_papers.csv", encoding="utf-8")))
    snapshot_missing = sum(1 for _ in csv.DictReader(open(SNAPSHOT_MISSING, encoding="utf-8")))
    added = added_latex()

    latest = sorted(glob.glob(os.path.join(ITER_DIR, "iter_*", "hits.jsonl")))[-1]
    flagged, blocks = set(), 0
    for path in [latest] + [ADDED_HITS] * os.path.exists(ADDED_HITS):
        for line in open(path):
            d = json.loads(line)
            flagged.add(d["id"])
            blocks += count_blocks(d["hits"])
    # The development rounds and the gold standard sampled the snapshot only.
    flagged_snapshot = flagged - added

    labels, rounds = {}, set()
    for f in glob.glob(os.path.join(ITER_DIR, "iter_*", "review_db", "**", "labels", "*.json"), recursive=True):
        d = json.load(open(f))
        if d.get("label") in ("prompt", "no_prompt"):
            labels[d["paper_id"]] = d["label"] == "prompt"
            rounds.add(f.split(os.sep)[3])
    # The review rounds sample flagged and unflagged papers equally, but unflagged
    # papers are most of the population, so recall weights each stratum by its size.
    snapshot = listed - snapshot_missing
    searched = snapshot + len(added)
    tp = sum(1 for pid, pos in labels.items() if pos and pid in flagged_snapshot)
    fp = sum(1 for pid, pos in labels.items() if not pos and pid in flagged_snapshot)
    fn = sum(1 for pid, pos in labels.items() if pos and pid not in flagged_snapshot)
    tn = sum(1 for pid, pos in labels.items() if not pos and pid not in flagged_snapshot)
    n_flagged, n_unflagged = len(flagged_snapshot), snapshot - len(flagged_snapshot)
    recall = lambda prec, miss: n_flagged * prec / (n_flagged * prec + n_unflagged * miss)
    prec, miss = tp / (tp + fp), fn / (fn + tn)
    (plo, phi), (mlo, mhi) = wilson(tp, tp + fp), wilson(fn, fn + tn)
    # Conservative interval: combine the ends of both stratum intervals.
    rec, rlo, rhi = recall(prec, miss), recall(plo, mhi), recall(phi, mlo)

    pct = lambda x: f"{100 * x:.0f}\\%"
    num = lambda x: f"{x:,}".replace(",", "{,}")
    values = {
        "numListed": num(listed),
        "numLatex": num(searched),
        "numMissing": num(listed - searched),
        "numSnapshotLatex": num(snapshot),
        "numAddedLatex": num(len(added)),
        "numPatterns": str(len(collect_prompts.PATTERNS)),
        "numFlagged": num(len(flagged)),
        "numPrompts": num(blocks),
        "numRounds": str(len(rounds)),
        "numLabeled": num(len(labels)),
        "numRoundSample": str(collect_prompts.SAMPLE_SIZE),
        "devPrecision": pct(prec),
        "devPrecisionCI": f"{pct(plo)}--{pct(phi)}",
        "devRecall": pct(rec),
        "devRecallCI": f"{pct(rlo)}--{pct(rhi)}",
        "confLevel": CONFIDENCE + "\\%",
        "goldMargin": f"{100 * GOLD_MARGIN:.0f}\\%",
        "goldSample": str(sample_size(snapshot, GOLD_MARGIN)),
        "goldSampleFive": str(sample_size(snapshot, 0.05)),
        "goldSampleSeven": str(sample_size(snapshot, 0.07)),
        "goldSampleTen": str(sample_size(snapshot, 0.10)),
    }
    if os.path.isdir(gold_agreement.ANNOTATIONS):
        g = gold_agreement.summary()
        values.update({
            "goldPapers": str(g["papers"]),
            "goldAgree": str(g["agree"]),
            "goldDisagree": str(len(g["disagreements"])),
            "goldKappa": f"{g['kappa']:.2f}",
            "goldAlpha": f"{g['alpha']:.2f}",
        })
        # Detector scores only once every disagreement is resolved in data/gold/adjudicated.csv.
        if not g["unresolved"]:
            (plo, phi), (rlo, rhi) = g["precision_ci"], g["recall_ci"]
            values.update({
                "goldPositives": str(g["gold_positives"]),
                "goldPrecision": pct(g["precision"]),
                "goldPrecisionCI": f"{pct(plo)}--{pct(phi)}",
                "goldRecall": pct(g["recall"]),
                "goldRecallCI": f"{pct(rlo)}--{pct(rhi)}",
            })
            # Gold papers the detector did not flag, and how many of them have prompts anyway.
            mlo, mhi = wilson(g["fn"], g["unflagged"])
            values.update({
                "goldUnflagged": str(g["unflagged"]),
                "goldMissed": str(g["fn"]),
                "goldMissedShare": pct(g["fn"] / g["unflagged"]),
                "goldMissedShareCI": f"{pct(mlo)}--{pct(mhi)}",
                "goldMissedText": str(g["missed_text"]),
            })
        else:
            print(f"gold: {len(g['unresolved'])} disagreements unresolved, detector scores left out")
        # Prompt level, once the annotators' prompt lists are reconciled.
        pl = g["prompt_level"]
        if pl:
            (rlo, rhi), (elo, ehi) = pl["recall_ci"], pl["estimated_ci"]
            values.update({
                "goldPrompts": str(pl["prompts"]),
                "goldPromptsFound": str(pl["found"]),
                "goldPromptRecall": pct(pl["recall"]),
                "goldPromptRecallCI": f"{pct(rlo)}--{pct(rhi)}",
                "goldPromptsBoth": str(pl["both"]),
                "goldPromptsOnlyA": str(pl["by_a"] - pl["both"]),
                "goldPromptsOnlyB": str(pl["by_b"] - pl["both"]),
                "goldPromptsEstimated": f"{pl['estimated']:.0f}",
                "goldPromptsEstimatedCI": f"{elo:.0f}--{ehi:.0f}",
                "goldPromptsEstimatedByWhere": f"{pl['estimated_by_where']:.0f}",
                "goldCoverageA": pct(pl["by_a"] / pl["prompts"]),
                "goldCoverageB": pct(pl["by_b"] / pl["prompts"]),
            })
    values.update(cluster_numbers(blocks, pct))
    values.update(task_numbers(pct))
    values.update(assign_numbers(pct))
    values.update(dataset_numbers(pct))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write(f"% Generated by paper_numbers.py from {latest}; do not edit by hand.\n")
        for name, v in values.items():
            f.write(f"\\providecommand{{\\{name}}}{{}}\\renewcommand{{\\{name}}}{{{v}}}\n")
    print(json.dumps(values, indent=1))


if __name__ == "__main__":
    main()
