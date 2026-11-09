"""Cluster the extracted prompt blocks by SE task (stage 3 of the approach figure).

Reads the prompt blocks of the latest collect_prompts.py iteration (overlapping hits
in a paper are one block, as paper_numbers.count_blocks counts them). Blocks found not
to be prompts are listed in data/prompts/block_review.csv (block_id, keep, source),
by the block review or by import-screening. They are still embedded and clustered, so
a rejection never moves the other blocks or undoes a cluster's screening, but they
are left out of every count and sample. So are the blocks of clusters whose screening
found mostly non-prompts (DROP_SHARE).

  cluster     normalise each block (LaTeX stripped, placeholders masked), embed it,
              group near-duplicates into families, and cluster one representative
              per family with BERTopic (UMAP + HDBSCAN); every block takes its
              family's cluster. Manual edits from data/clusters/edits.json are then
              applied in order (see apply_edits). Writes data/clusters/:
                blocks.jsonl   every block: id, paper, family, cluster, normalised text
                clusters.json  per cluster: size, families, papers, top terms, examples,
                               and its name from NAMES while its members are unchanged
                summary.json   counts read by paper_numbers.py, the unassigned share at
                               other HDBSCAN settings, and the run's model and settings
  assign      place the blocks of papers added after the clusters were frozen
              (ADDED_HITS) in the existing clusters by their nearest clustered blocks;
              appends them to blocks.jsonl and writes assigned.json (see assign).
              `cluster` rewrites blocks.jsonl, so run `assign` again after it.
  second-pass group the prompts still unassigned (UMAP + HDBSCAN, leaf selection), apply
              the ops in data/clusters/second_pass.json, and keep the groups it names
              as clusters, mapped to types in task_types.json (see second_pass). Run
              it after every `assign`.
  screen      write the next screening round, data/clusters/screening/round_NN.csv,
              for every cluster whose members have not been screened yet (a result
              holds while the cluster is unchanged, so later rounds only cover merged,
              split or otherwise changed clusters): SCREEN_FIRST + SCREEN_SECOND +
              SCREEN_RESERVE prompts at most one per paper, none labelled before, in a
              random `order`. A coder fills `fits` in that order (1 fits, 0 misfit,
              x not a prompt) and stops once the outcome is settled (screen_outcome);
              a non-prompt is skipped and the next one takes its place. A cluster to
              revise is merged or split in edits.json.
  import-screening ROUND_CSV EXPORT_DIR
              fill `fits` in a round from the labels exported from the annotation
              page, and list its non-prompts in BLOCK_REVIEW.
  import-validation LABELS_DIR RESOLVED_DIR
              write data/clusters/validation.csv from the review page's exported
              labels (one file per coder and prompt) and resolutions; coder a is the
              one who labelled first.
  import-names EXPORT_DIR
              record the cluster names edited on the annotation page in NAMES.
  validation  once clusters and task types are frozen, write
              data/clusters/validation_sample.csv: a sample of every prompt (added papers
              included) sized for CLUSTER_MARGIN, stratified by the major group of its
              task type (review_groups; unassigned prompts and those of REVIEW_NONE_GROUP
              are one stratum) with proportional allocation, at most one prompt per paper
              per group, and no prompt screened before. Both coders fill fit_a and fit_b,
              resolve disagreements in fit, and save it as data/clusters/validation.csv.
              For a prompt in no group, 1 means it rightly fits none of the groups; x
              marks a block that is not a prompt.

Runs in the container from docker/cluster.Dockerfile:
  docker run --rm -u $(id -u):$(id -g) -v "$PWD":/work -w /work se-prompts-cluster \\
      python task_clusters.py cluster|screen|validation
"""
import csv
import glob
import hashlib
import json
import os
import random
import re
import sys

import paper_numbers

ITER_DIR = "data/prompts/iterations"
BLOCK_REVIEW = "data/prompts/block_review.csv"
# Cluster names by members fingerprint ({"cluster", "name"}): from
# the top terms and examples, or set by a coder on the annotation page.
NAMES = "data/clusters/names.json"
OUT_DIR = "data/clusters"
SEED = 2026
# The best open model under 1B parameters, zero-shot on the clustering tasks of
# MTEB(eng, v2), at the revision MTEB scored.
EMBEDDING_MODEL = "microsoft/harrier-oss-v1-0.6b"
EMBEDDING_REVISION = "f9b9dc8d367d443f2479d27aa5d8d2850c0774ee"
# Instruction in the form and template MTEB uses for this model on clustering tasks
# ("Identify the main category of Biorxiv papers based on the titles and abstracts").
INSTRUCTION = "Instruct: Identify the software engineering task of prompts written for large language models\nQuery:"
# Tokens embedded per prompt; longer prompts are cut (about one block in fifty).
MAX_TOKENS = 1024
# Blocks with fewer words after normalisation (e.g., a box around an image) are not clustered.
MIN_WORDS = 5
# Blocks whose normalised embeddings are at least this similar are one family
# (variants of one template, or a template reused across papers).
DUP_THRESHOLD = 0.95
# HDBSCAN settings tried, as (min_cluster_size in families, min_samples; None means
# equal to min_cluster_size). The run uses the one with the highest DBCV (HDBSCAN's
# relative_validity_), a density-based validity index that needs no labels; every
# setting's cluster count and unassigned share go to summary.json.
HDBSCAN_GRID = [(size, samples) for size in (10, 15, 20, 30) for samples in (None, 1, 3, 5, 10)]
UMAP_PARAMS = dict(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine")
EXAMPLES = 5
SCREEN_TEXT_CHARS = 1500
# A cluster is dropped as an extraction error when at least DROP_SHARE of its labelled
# blocks (all rounds, at least DROP_MIN_LABELLED) are not prompts; its blocks leave the
# dataset with it.
DROP_SHARE = 0.5
DROP_MIN_LABELLED = 5
# Clusters left with fewer families after the edits (small parts of a split) are dissolved.
MIN_EDITED_FAMILIES = 5
# Extra prompts per cluster in a screening round, to replace those that are not prompts.
SCREEN_RESERVE = 5
# Prompts from papers added after the clusters were frozen (collect_prompts.py --hits-only)
# and the totals over every block once they are assigned.
ADDED_HITS = "data/prompts/added/hits.jsonl"
ASSIGNED = os.path.join(OUT_DIR, "assigned.json")
# An added block takes the cluster most of its ASSIGN_K nearest clustered blocks are in
# (the smallest HDBSCAN cluster size tried, so a vote can rest on one small cluster).
ASSIGN_K = 10
# The task types: the named clusters grouped by hand into types, and the types into SE
# activities ({"groups": [{"group", "types": [{"name", "clusters": [cluster names],
# "second_pass": [second-pass group names]}]}]}).
TASK_TYPES = os.path.join(OUT_DIR, "task_types.json")
# Second pass over the prompts still unassigned after `assign`: its HDBSCAN setting, the
# split/dissolve ops on its groups, and the groups kept by name ({"ops", "groups": {id: name}}).
# Kept groups become clusters with ids from SECOND_PASS_BASE; the others stay unassigned.
SECOND_PASS = os.path.join(OUT_DIR, "second_pass.json")
SECOND_PASS_BASE = 10000
# Leaf selection: the second pass looks for the small, dense groups the first pass's
# excess-of-mass selection left out.
SECOND_PASS_HDBSCAN = dict(min_cluster_size=10, min_samples=5, cluster_selection_method="leaf")
# The validation labels the major groups of TASK_TYPES (the SE activities and the
# LLM-centric tasks); the catch-all group is reviewed with the unassigned prompts.
REVIEW_NONE_GROUP = paper_numbers.REVIEW_NONE_GROUP

# --- LaTeX -> prose --------------------------------------------------------
BRACED = r"\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}"
# commands whose arguments are not prompt text
DROP_WITH_ARGS = re.compile(
    r"\\(?:caption|captionof|label|ref|autoref|cref|Cref|eqref|cite[tpn]?|url|href|"
    r"includegraphics|vspace|hspace|setlength|setcounter|addtocounter|definecolor|color|"
    r"cellcolor|rowcolor|rule|newcommand|renewcommand|providecommand|newtcolorbox|"
    r"newenvironment|tcbset|lstset|lstdefinestyle)\*?(?:\[[^\]]*\])?"
    rf"(?:\s*{BRACED})*")
# commands whose leading arguments are layout and whose last argument is text
LAYOUT_ARGS = {"resizebox": 2, "scalebox": 1, "textcolor": 1, "colorbox": 1, "fcolorbox": 2,
               "multirow": 2, "multicolumn": 2, "tabincell": 1, "parbox": 1, "makebox": 0}
DROP_LAYOUT_ARGS = re.compile("|".join(
    rf"\\{name}\*?(?:\[[^\]]*\])*" + rf"(?:\s*{BRACED}){{{n}}}" for name, n in LAYOUT_ARGS.items()))
# length settings such as \aboverulesep=0ex
DROP_ASSIGNMENT = re.compile(r"\\[A-Za-z@]+\s*=\s*-?[\d.]+\s*[a-z]{0,2}")
ENV_TAG = re.compile(rf"\\(?:begin|end)\{{[^}}]*\}}(?:\[[^\]]*\]|{BRACED})*")
COMMAND = re.compile(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?")
ESCAPED = {r"\{": "\x01", r"\}": "\x02", r"\_": "_", r"\#": "#", r"\%": "%",
           r"\&": "&", r"\$": "\x03", r"\textbackslash": "\\"}
# {code}, {{input}}, <issue>, [INPUT], ${var}: template slots, filled differently in each paper
SLOT = re.compile(r"\{\{?\s*#?[A-Za-z_][\w .\-/]{0,40}\}\}?|<\s*[A-Za-z_][\w .\-/]{0,30}>"
                  r"|\[[A-Z_][A-Z0-9_ ]{1,30}\]|\$\{\w+\}")
SPACE = re.compile(r"[ \t]+")
BLANK_LINES = re.compile(r"\n\s*\n+")


def normalise(text):
    for k, v in ESCAPED.items():
        text = text.replace(k, v)
    text = DROP_WITH_ARGS.sub(" ", text)
    text = DROP_LAYOUT_ARGS.sub(" ", text)
    text = DROP_ASSIGNMENT.sub(" ", text)
    text = ENV_TAG.sub("\n", text)
    text = text.replace("\\\\", "\n")
    text = COMMAND.sub(" ", text)
    text = re.sub(r"[{}&~$]", " ", text)
    text = text.replace("\x01", "{").replace("\x02", "}").replace("\x03", "$")
    text = SLOT.sub(" [SLOT] ", text)
    text = SPACE.sub(" ", text)
    return BLANK_LINES.sub("\n", "\n".join(line.strip() for line in text.splitlines())).strip()


# --- blocks ----------------------------------------------------------------
def latest_hits():
    return sorted(glob.glob(os.path.join(ITER_DIR, "iter_*", "hits.jsonl")))[-1]


def not_prompts():
    """Block ids found not to be prompts, from BLOCK_REVIEW."""
    if not os.path.exists(BLOCK_REVIEW):
        return set()
    return {r["block_id"] for r in csv.DictReader(open(BLOCK_REVIEW, encoding="utf-8")) if r["keep"].strip() == "0"}


def cluster_names():
    return json.load(open(NAMES)) if os.path.exists(NAMES) else {}


def task_types():
    """{cluster: {"type", "name", "group"}} from TASK_TYPES, matching clusters by name.

    Type ids count up in the file's order. Every named cluster must be in exactly one
    type, and every cluster the file names must exist.
    """
    spec = json.load(open(TASK_TYPES))
    clusters = [c for c in json.load(open(os.path.join(OUT_DIR, "clusters.json"))) if c["cluster"] >= 0]
    by_name = {}
    for c in clusters:
        if c["name"] in by_name:
            sys.exit(f"two clusters are named {c['name']!r}")
        by_name[c["name"]] = c["cluster"]
    types, type_id = {}, 0
    for g in spec["groups"]:
        for t in g["types"]:
            for name in t["clusters"] + t.get("second_pass", []):
                if name not in by_name:
                    sys.exit(f"{TASK_TYPES}: no cluster is named {name!r}")
                if by_name[name] in types:
                    sys.exit(f"{TASK_TYPES}: {name!r} is in two types")
                types[by_name[name]] = {"type": type_id, "name": t["name"], "group": g["group"]}
            type_id += 1
    missing = [c["name"] or c["cluster"] for c in clusters if c["cluster"] not in types]
    if missing:
        sys.exit(f"{TASK_TYPES}: clusters in no type: {missing}")
    return types


def merge_hits(hits):
    """One paper's hits ({"method", "start", "end", "text"}), overlapping ones stitched into
    blocks, in order of position. start/end are offsets into the paper's comment-stripped
    TeX (collect_prompts.strip_comments)."""
    merged = []
    for h in sorted(hits, key=lambda h: h["start"]):
        if merged and h["start"] < merged[-1]["end"]:
            last = merged[-1]
            last["methods"].add(h["method"])
            if h["end"] > last["end"]:
                last["text"] += h["text"][last["end"] - h["start"]:]
                last["end"] = h["end"]
        else:
            merged.append({"start": h["start"], "end": h["end"], "text": h["text"], "methods": {h["method"]}})
    return [{"raw": b["text"], "start": b["start"], "end": b["end"], "methods": sorted(b["methods"])} for b in merged]


def load_blocks(path):
    """Prompt blocks: overlapping hits in a paper are stitched into one block."""
    blocks = []
    for line in open(path):
        d = json.loads(line)
        for i, b in enumerate(merge_hits(d["hits"])):
            blocks.append({"block_id": f"{d['id']}#{i}", "paper_id": d["id"], **b})
    return blocks


# --- embeddings and families ---------------------------------------------------
def embed(texts, model, name="embeddings"):
    """Embeddings, cached on disk (OUT_DIR/name.npy) for the same model and texts."""
    import numpy as np
    key = hashlib.sha256("\x00".join([EMBEDDING_MODEL, EMBEDDING_REVISION, INSTRUCTION, str(MAX_TOKENS), *texts])
                         .encode()).hexdigest()
    path = os.path.join(OUT_DIR, f"{name}.npy")
    key_path = path + ".key"
    if os.path.exists(path) and open(key_path).read() == key:
        return np.load(path)
    vectors = model.encode(texts, prompt=INSTRUCTION, batch_size=8, normalize_embeddings=True,
                           show_progress_bar=True)
    np.save(path, vectors)
    open(key_path, "w").write(key)
    return vectors


def families(vectors):
    """Union-find over pairs at least DUP_THRESHOLD similar; returns family per block and representatives."""
    import numpy as np
    parent = list(range(len(vectors)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for start in range(0, len(vectors), 1000):
        sims = vectors[start:start + 1000] @ vectors.T
        for i, j in zip(*np.nonzero(sims >= DUP_THRESHOLD)):
            a, b = find(start + i), find(j)
            if a != b:
                parent[max(a, b)] = min(a, b)
    roots = [find(i) for i in range(len(vectors))]
    members = {}
    for i, r in enumerate(roots):
        members.setdefault(r, []).append(i)
    # Families are numbered in block order; each is represented by its medoid.
    family_of, reps = [0] * len(vectors), []
    for f, (_, idx) in enumerate(sorted(members.items())):
        for i in idx:
            family_of[i] = f
        sub = vectors[idx]
        reps.append(idx[int(np.argmax((sub @ sub.T).sum(axis=1)))])
    return family_of, reps


# --- clustering ------------------------------------------------------------
def hdbscan_model(min_cluster_size, min_samples=None):
    from hdbscan import HDBSCAN
    return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean",
                   cluster_selection_method="eom", gen_min_span_tree=True)


def choose_hdbscan(reduced, family_of):
    """Fit every HDBSCAN_GRID setting; return the best by DBCV and every setting's outcome.

    family_of lists the family of each block counted as unassigned (the prompts)."""
    import numpy as np
    family_of = np.array(family_of)
    grid = []
    for size, samples in HDBSCAN_GRID:
        h = hdbscan_model(size, samples).fit(reduced)
        grid.append({"min_cluster_size": size, "min_samples": samples, "clusters": int(h.labels_.max()) + 1,
                     "unassigned": int((h.labels_[family_of] < 0).sum()), "dbcv": round(float(h.relative_validity_), 4)})
    best = max(grid, key=lambda g: g["dbcv"])
    return best, grid


def apply_edits(labels, reduced, vectors, min_cluster_size):
    """Apply data/clusters/edits.json to the representatives' cluster labels.

    {"ops": [{"merge": [3, 7]}, {"split": 5}, {"dissolve": 9}, ...]}. The ops
    run in order, each on the ids left by the ones before. A merge moves every
    cluster listed into the smallest id. A split re-runs HDBSCAN (half the minimum
    size) on the cluster's points in the UMAP space; if that finds no sub-clusters,
    k-means (k=2) on their embeddings splits it in two. The largest part keeps the
    id, the others take new ids after the largest ever used, and points the split
    leaves unassigned stay in the cluster. A dissolve makes the cluster's points
    unassigned (a cluster with no common task). Clusters left with fewer than
    MIN_EDITED_FAMILIES families are dissolved too. Ids are never renumbered, so the edits keep
    pointing at the same clusters when the run is repeated.
    """
    import numpy as np
    path = os.path.join(OUT_DIR, "edits.json")
    edits = json.load(open(path)) if os.path.exists(path) else {}
    labels = np.array(labels)
    # New ids count up from the largest id ever used, so a merge never frees an id.
    next_id = labels.max() + 1
    for op in edits.get("ops", []):
        if "merge" in op:
            ids = sorted(op["merge"])
            labels[np.isin(labels, ids)] = ids[0]
        elif "dissolve" in op:
            labels[labels == op["dissolve"]] = -1
        elif "split" in op:
            idx = np.nonzero(labels == op["split"])[0]
            sub = hdbscan_model(max(5, min_cluster_size // 2)).fit(reduced[idx]).labels_
            parts = sorted({p for p in sub if p >= 0}, key=lambda p: -(sub == p).sum())
            if len(parts) < 2:
                from sklearn.cluster import KMeans
                sub = KMeans(n_clusters=2, random_state=SEED, n_init=10).fit(vectors[idx]).labels_
                parts = sorted({0, 1}, key=lambda p: -(sub == p).sum())
                print(f"split {op['split']}: HDBSCAN found no sub-clusters, used k-means (k=2)")
            for p in parts[1:]:
                labels[idx[sub == p]] = next_id
                next_id += 1
    for c in set(labels.tolist()) - {-1}:
        if (labels == c).sum() < MIN_EDITED_FAMILIES:
            labels[labels == c] = -1
    return labels.tolist()


# --- screening results ---------------------------------------------------------
def members_key(block_ids):
    """Fingerprint of a cluster's members; a screening result holds while it is unchanged."""
    return hashlib.sha256("\n".join(sorted(block_ids)).encode()).hexdigest()[:12]


def screen_outcome(rows):
    """accept, revise or open, from one cluster's rows (order, fits) of one round.

    Non-prompts (x) are skipped; the first SCREEN_FIRST other prompts in order are
    stage 1 and the next SCREEN_SECOND stage 2. Accept if every stage-1 prompt fits,
    or exactly one does not and every stage-2 prompt fits; revise at the second
    misfit. A cluster with too few prompts for a stage is judged on those it has.
    """
    judged = [r["fits"].strip() for r in sorted(rows, key=lambda r: int(r["order"])) if r["fits"].strip() != "x"]
    first = judged[:paper_numbers.SCREEN_FIRST]
    second = judged[paper_numbers.SCREEN_FIRST:paper_numbers.SCREEN_FIRST + paper_numbers.SCREEN_SECOND]
    misfits = first.count("0") + second.count("0")
    # Two misfits, or a sample of nothing but non-prompts.
    if misfits >= 2 or not first:
        return "revise"
    if "" in first:
        return "open"
    if "0" not in first:
        return "accept"
    return "open" if "" in second else "accept"


def screen_labels():
    """Every label a coder gave (1, 0 or x) per block_id, over all rounds."""
    return {r["block_id"]: r["fits"].strip()
            for path in sorted(glob.glob(os.path.join(OUT_DIR, "screening", "round_*.csv")))
            for r in csv.DictReader(open(path, encoding="utf-8")) if r["fits"].strip()}


def dropped_clusters(cluster_of, blocks):
    """Clusters whose labelled blocks are mostly not prompts (DROP_SHARE)."""
    labels = screen_labels()
    seen = {}
    for c, b in zip(cluster_of, blocks):
        if c >= 0 and b["block_id"] in labels:
            seen.setdefault(c, []).append(labels[b["block_id"]])
    return {c for c, ls in seen.items()
            if len(ls) >= DROP_MIN_LABELLED and ls.count("x") / len(ls) >= DROP_SHARE}


def screening_results():
    """Outcome per cluster fingerprint over all rounds; a later round overrides an earlier one."""
    results = {}
    for path in sorted(glob.glob(os.path.join(OUT_DIR, "screening", "round_*.csv"))):
        groups = {}
        for r in csv.DictReader(open(path, encoding="utf-8")):
            groups.setdefault(r["members"], []).append(r)
        for key, rows in groups.items():
            results[key] = screen_outcome(rows)
    return results


def cluster():
    import numpy as np
    import bertopic
    import sklearn
    from bertopic import BERTopic
    from bertopic.dimensionality import BaseDimensionalityReduction
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
    from umap import UMAP

    os.makedirs(OUT_DIR, exist_ok=True)
    hits = latest_hits()
    blocks = load_blocks(hits)
    texts = [normalise(b["raw"]) for b in blocks]
    keep = [i for i, t in enumerate(texts) if len(t.split()) >= MIN_WORDS]
    too_short = len(blocks) - len(keep)
    blocks, texts = [blocks[i] for i in keep], [texts[i] for i in keep]
    model = SentenceTransformer(EMBEDDING_MODEL, revision=EMBEDDING_REVISION, model_kwargs={"dtype": "float32"})
    model.max_seq_length = MAX_TOKENS
    vectors = embed(texts, model)
    family_of, reps = families(vectors)
    rejected = not_prompts()
    is_prompt = [b["block_id"] not in rejected for b in blocks]
    print(f"{len(blocks)} blocks ({is_prompt.count(False)} not prompts, {too_short} too short), {len(reps)} families")

    # UMAP once, so the HDBSCAN settings are compared and BERTopic clusters in the same space.
    rep_texts = [texts[i] for i in reps]
    reduced = UMAP(random_state=SEED, **UMAP_PARAMS).fit_transform(vectors[reps])
    best, grid = choose_hdbscan(reduced, [f for f, p in zip(family_of, is_prompt) if p])
    print(f"HDBSCAN min_cluster_size={best['min_cluster_size']}, min_samples={best['min_samples']} (DBCV {best['dbcv']})")
    topic_model = BERTopic(
        embedding_model=model,
        umap_model=BaseDimensionalityReduction(),
        hdbscan_model=hdbscan_model(best["min_cluster_size"], best["min_samples"]),
        vectorizer_model=CountVectorizer(stop_words=sorted(ENGLISH_STOP_WORDS | {"slot"}), min_df=2,
                                         ngram_range=(1, 2),
                                         token_pattern=r"(?u)\b[A-Za-z][A-Za-z]+\b"),
    )
    labels, _ = topic_model.fit_transform(rep_texts, reduced)
    labels = apply_edits(labels, reduced, vectors[reps], best["min_cluster_size"])
    topic_model.update_topics(rep_texts, topics=labels)

    cluster_of = [labels[f] for f in family_of]
    dropped = dropped_clusters(cluster_of, blocks)
    is_dropped = [c in dropped for c in cluster_of]
    # Blocks not yet marked as non-prompts that leave with their cluster.
    n_dropped = sum(p and d for p, d in zip(is_prompt, is_dropped))
    is_prompt = [p and not d for p, d in zip(is_prompt, is_dropped)]
    print(f"dropped clusters (mostly not prompts): {sorted(dropped)}, {n_dropped} more blocks")
    papers_per_family = {}
    for b, f, p in zip(blocks, family_of, is_prompt):
        if p:
            papers_per_family.setdefault(f, set()).add(b["paper_id"])
    with open(os.path.join(OUT_DIR, "blocks.jsonl"), "w") as out:
        for b, t, f, c, p, d in zip(blocks, texts, family_of, cluster_of, is_prompt, is_dropped):
            out.write(json.dumps({"block_id": b["block_id"], "paper_id": b["paper_id"], "family": f,
                                  "family_papers": len(papers_per_family.get(f, ())), "cluster": c,
                                  "prompt": p, "dropped": d, "text": t}, ensure_ascii=False) + "\n")

    rng = random.Random(SEED)
    results = screening_results()
    names = cluster_names()
    clusters = []
    for c in sorted(set(cluster_of)):
        members = [i for i, x in enumerate(cluster_of) if x == c]
        idx = [i for i in members if is_prompt[i]]
        if not idx:
            continue
        terms = [w for w, _ in topic_model.get_topic(c) or []][:10]
        # Over every member, so marking a non-prompt keeps the cluster's screening.
        key = members_key([blocks[i]["block_id"] for i in members])
        clusters.append({
            "cluster": c, "members": key, "name": names.get(key, {}).get("name"),
            "screening": results.get(key), "blocks": len(idx),
            "families": len({family_of[i] for i in idx}), "papers": len({blocks[i]["paper_id"] for i in idx}),
            "terms": terms,
            "examples": [{"block_id": blocks[i]["block_id"], "text": texts[i][:400]}
                         for i in rng.sample(idx, min(EXAMPLES, len(idx)))],
        })
    json.dump(clusters, open(os.path.join(OUT_DIR, "clusters.json"), "w"), indent=1, ensure_ascii=False)

    summary = {
        "prompts": is_prompt.count(True), "not_prompts": len(rejected & {b["block_id"] for b in blocks}),
        "dropped_clusters": len(dropped), "dropped": n_dropped, "too_short": too_short,
        "families": len(papers_per_family),
        "reused_families": sum(1 for p in papers_per_family.values() if len(p) > 1),
        "clusters": len([c for c in clusters if c["cluster"] >= 0]),
        "unassigned": sum(1 for c, p in zip(cluster_of, is_prompt) if c < 0 and p),
        "hdbscan": best, "hdbscan_grid": grid,
        "input": hits, "seed": SEED, "embedding_model": EMBEDDING_MODEL,
        "embedding_revision": EMBEDDING_REVISION, "instruction": INSTRUCTION, "max_tokens": MAX_TOKENS,
        "min_words": MIN_WORDS, "dup_threshold": DUP_THRESHOLD, "umap": UMAP_PARAMS,
        "versions": {"bertopic": bertopic.__version__, "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__},
    }
    json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("umap", "versions", "hdbscan_grid")}, indent=1))


# --- papers added after the clusters were frozen -----------------------------------
def knn_vote(sims, labels, k):
    """Per row of `sims` (similarities to the labelled blocks), the label most of the k
    most similar blocks carry; a tie goes to the label with the larger summed similarity."""
    import numpy as np
    top = np.argsort(-sims, axis=1)[:, :k]
    votes = []
    for row, idx in zip(sims, top):
        score = {}
        for i in idx:
            n, s = score.get(labels[i], (0, 0.0))
            score[labels[i]] = (n + 1, s + float(row[i]))
        votes.append(max(score, key=score.get))
    return votes


def assign():
    """Place the blocks of ADDED_HITS in the frozen clusters, without re-clustering.

    An added block whose embedding is at least DUP_THRESHOLD similar to a clustered
    block joins that block's family and cluster. Any other block takes the cluster (or
    unassigned) most of its ASSIGN_K nearest clustered prompts are in; blocks of dropped
    clusters vote too, so a block that lands among them is dropped as well. Added blocks
    near-identical to each other form new families. The clusters, their screening and
    names stay as they are. Appends the added blocks to blocks.jsonl (marked "added")
    and writes ASSIGNED: totals over every block, and a leave-one-family-out check of
    the vote on the clustered prompts. Run again after every `cluster`.
    """
    import numpy as np
    from sentence_transformers import SentenceTransformer

    rows = [json.loads(line) for line in open(os.path.join(OUT_DIR, "blocks.jsonl"))]
    base = [r for r in rows if not r.get("added")]
    # The second pass runs after this one; undo it so it never votes.
    for r in base:
        if r.pop("second_pass", False):
            r["cluster"] = -1
    vectors = np.load(os.path.join(OUT_DIR, "embeddings.npy"))
    if len(vectors) != len(base):
        sys.exit("embeddings.npy does not match blocks.jsonl; run `cluster` first")
    known = {r["block_id"] for r in base}
    added = [b for b in load_blocks(ADDED_HITS) if b["block_id"] not in known]
    texts = [normalise(b["raw"]) for b in added]
    keep = [i for i, t in enumerate(texts) if len(t.split()) >= MIN_WORDS]
    too_short = len(added) - len(keep)
    added, texts = [added[i] for i in keep], [texts[i] for i in keep]
    model = SentenceTransformer(EMBEDDING_MODEL, revision=EMBEDDING_REVISION, model_kwargs={"dtype": "float32"})
    model.max_seq_length = MAX_TOKENS
    new = embed(texts, model, name="added_embeddings")

    # Voters: clustered prompts and the blocks of dropped clusters, not single rejected blocks.
    dropped = {r["cluster"] for r in base if r["dropped"]}
    voters = [i for i, r in enumerate(base) if r["prompt"] or r["dropped"]]
    labels = [base[i]["cluster"] for i in voters]
    sims = new @ vectors[voters].T
    votes = {k: knn_vote(sims, labels, k) for k in (5, ASSIGN_K, 20)}

    # Families: a group of near-identical added blocks joins the clustered family its
    # best member matches, if any; otherwise it is a new family.
    groups, _ = families(new) if len(new) else ([], [])
    best, best_sim = sims.argmax(axis=1), sims.max(axis=1)
    next_family = max(r["family"] for r in base) + 1
    family_of, joined = {}, {}
    for i, g in enumerate(groups):
        if best_sim[i] >= DUP_THRESHOLD and (g not in joined or best_sim[i] > best_sim[joined[g]]):
            joined[g] = i
    for g in sorted(set(groups)):
        if g in joined:
            family_of[g] = base[voters[best[joined[g]]]]["family"]
        else:
            family_of[g], next_family = next_family, next_family + 1
    cluster_of_family = {r["family"]: r["cluster"] for r in base}

    out = []
    for i, (b, t) in enumerate(zip(added, texts)):
        f = family_of[groups[i]]
        c = cluster_of_family[f] if groups[i] in joined else votes[ASSIGN_K][i]
        out.append({"block_id": b["block_id"], "paper_id": b["paper_id"], "family": f, "cluster": c,
                    "prompt": c not in dropped, "dropped": c in dropped, "added": True, "text": t})
    blocks = base + out
    papers_per_family = {}
    for r in blocks:
        if r["prompt"]:
            papers_per_family.setdefault(r["family"], set()).add(r["paper_id"])
    with open(os.path.join(OUT_DIR, "blocks.jsonl"), "w") as f:
        for r in blocks:
            r["family_papers"] = len(papers_per_family.get(r["family"], ()))
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # How well the vote reproduces the frozen clusters: each clustered prompt is voted on
    # by the other prompts, leaving out its own family (whose members share its cluster).
    prompts = [i for i, r in enumerate(base) if r["prompt"]]
    agree, fam = 0, np.array([base[i]["family"] for i in voters])
    for start in range(0, len(prompts), 1000):
        chunk = prompts[start:start + 1000]
        s = vectors[chunk] @ vectors[voters].T
        s[fam[None, :] == np.array([base[i]["family"] for i in chunk])[:, None]] = -np.inf
        agree += sum(v == base[i]["cluster"] for v, i in zip(knn_vote(s, labels, ASSIGN_K), chunk))
    voted = [i for i in range(len(out)) if groups[i] not in joined]
    same = lambda k: sum(votes[k][i] == votes[ASSIGN_K][i] for i in voted) / max(1, len(voted))
    totals = {
        "prompts": sum(r["prompt"] for r in blocks),
        "unassigned": sum(r["prompt"] and r["cluster"] < 0 for r in blocks),
        "papers": len({r["paper_id"] for r in blocks if r["prompt"]}),
        "families": len(papers_per_family),
        "reused_families": sum(1 for p in papers_per_family.values() if len(p) > 1),
        "added": {
            "hits": ADDED_HITS, "k": ASSIGN_K, "blocks": len(out), "too_short": too_short,
            "papers": len({o["paper_id"] for o in out if o["prompt"]}),
            "prompts": sum(o["prompt"] for o in out), "dropped": sum(o["dropped"] for o in out),
            "unassigned": sum(o["prompt"] and o["cluster"] < 0 for o in out),
            "joined_family": len(out) - len(voted),
            # Share of voted blocks that land in the same cluster with k = 5 or 20.
            "same_k5": same(5), "same_k20": same(20),
        },
        "vote_check": {"prompts": len(prompts), "agree": int(agree)},
    }
    json.dump(totals, open(ASSIGNED, "w"), indent=1)
    print(json.dumps(totals, indent=1))


def second_pass():
    """Group the prompts still unassigned after `assign` and keep the groups named in SECOND_PASS.

    One representative per family is reduced with UMAP and clustered with HDBSCAN
    (SECOND_PASS_HDBSCAN); the ops in SECOND_PASS then split or dissolve groups as
    apply_edits does (new ids count up from 1000), and groups left with fewer than
    MIN_EDITED_FAMILIES families are dissolved. Each kept group becomes a cluster
    (SECOND_PASS_BASE + its id) with its name, which task_types.json maps to a type;
    every other prompt stays unassigned. Rewrites blocks.jsonl (marking the blocks
    "second_pass") and appends the kept groups to clusters.json; writes
    second_pass_summary.json. Run again after every `assign`.
    """
    import numpy as np
    from hdbscan import HDBSCAN
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    rows = [json.loads(line) for line in open(os.path.join(OUT_DIR, "blocks.jsonl"))]
    for r in rows:
        if r.pop("second_pass", False):
            r["cluster"] = -1
    parts = [np.load(os.path.join(OUT_DIR, "embeddings.npy"))]
    if any(r.get("added") for r in rows):
        parts.append(np.load(os.path.join(OUT_DIR, "added_embeddings.npy")))
    vectors = np.concatenate(parts)
    if len(vectors) != len(rows):
        sys.exit("embeddings do not match blocks.jsonl; run `cluster` and `assign` first")
    spec = json.load(open(SECOND_PASS))

    unassigned = [i for i, r in enumerate(rows) if r["prompt"] and r["cluster"] < 0]
    reps, seen = [], set()
    for i in unassigned:
        if rows[i]["family"] not in seen:
            seen.add(rows[i]["family"])
            reps.append(i)
    reduced = UMAP(random_state=SEED, **UMAP_PARAMS).fit_transform(vectors[reps])
    labels = HDBSCAN(**SECOND_PASS_HDBSCAN).fit(reduced).labels_.copy()
    found = int(labels.max()) + 1
    next_id = max(1000, found)
    for op in spec["ops"]:
        if "dissolve" in op:
            labels[labels == op["dissolve"]] = -1
        elif "split" in op:
            idx = np.nonzero(labels == op["split"])[0]
            sub = hdbscan_model(max(5, SECOND_PASS_HDBSCAN["min_cluster_size"] // 2)).fit(reduced[idx]).labels_
            pieces = sorted({p for p in sub if p >= 0}, key=lambda p: -(sub == p).sum())
            if len(pieces) < 2:
                sub = KMeans(n_clusters=op.get("k", 2), random_state=SEED, n_init=10).fit(vectors[reps][idx]).labels_
                pieces = sorted(set(sub), key=lambda p: -(sub == p).sum())
            for p in pieces[1:]:
                labels[idx[sub == p]] = next_id
                next_id += 1
    # One representative per family, so a group's size is its number of families.
    for g in set(labels.tolist()) - {-1}:
        if (labels == g).sum() < MIN_EDITED_FAMILIES:
            labels[labels == g] = -1
    kept = {int(g): name for g, name in spec["groups"].items()}
    gone = sorted(set(kept) - set(labels.tolist()))
    if gone:
        sys.exit(f"{SECOND_PASS}: groups {gone} do not exist")
    group_of_family = {rows[i]["family"]: int(g) for i, g in zip(reps, labels)}
    members = {}
    for i in unassigned:
        g = group_of_family[rows[i]["family"]]
        if g in kept:
            rows[i]["cluster"], rows[i]["second_pass"] = SECOND_PASS_BASE + g, True
            members.setdefault(g, []).append(i)
    with open(os.path.join(OUT_DIR, "blocks.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Distinctive terms per kept group (class-based TF-IDF over the kept groups).
    ids = sorted(members)
    docs = [" ".join(rows[i]["text"] for i in members[g]) for g in ids]
    cv = CountVectorizer(stop_words="english", token_pattern=r"(?u)\b[A-Za-z][A-Za-z]+\b")
    counts = cv.fit_transform(docs).toarray().astype(float)
    ctfidf = counts / counts.sum(1, keepdims=True) * np.log(1 + counts.sum(1).mean() / counts.sum(0))
    vocab = cv.get_feature_names_out()
    rng = random.Random(SEED)
    clusters = [c for c in json.load(open(os.path.join(OUT_DIR, "clusters.json"))) if not c.get("second_pass")]
    for g, row in zip(ids, ctfidf):
        idx = members[g]
        clusters.append({
            "cluster": SECOND_PASS_BASE + g, "members": members_key([rows[i]["block_id"] for i in idx]),
            "name": kept[g], "screening": None, "second_pass": True, "blocks": len(idx),
            "families": len({rows[i]["family"] for i in idx}), "papers": len({rows[i]["paper_id"] for i in idx}),
            "terms": [str(vocab[j]) for j in np.argsort(-row)[:10]],
            "examples": [{"block_id": rows[i]["block_id"], "text": rows[i]["text"][:400]}
                         for i in rng.sample(idx, min(EXAMPLES, len(idx)))],
        })
    json.dump(clusters, open(os.path.join(OUT_DIR, "clusters.json"), "w"), indent=1, ensure_ascii=False)
    prompts = [r for r in rows if r["prompt"]]
    summary = {
        "prompts": len(prompts), "unassigned_before": len(unassigned),
        "unassigned": sum(r["cluster"] < 0 for r in prompts),
        "groups_found": found, "groups": len(set(labels.tolist()) - {-1}), "kept": len(kept),
        "assigned": sum(len(v) for v in members.values()),
        "splits": sum("split" in o for o in spec["ops"]), "dissolves": sum("dissolve" in o for o in spec["ops"]),
        "hdbscan": SECOND_PASS_HDBSCAN, "min_families": MIN_EDITED_FAMILIES,
    }
    json.dump(summary, open(os.path.join(OUT_DIR, "second_pass_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


# --- samples for the coders ------------------------------------------------------
def load_clustered():
    blocks = [b for line in open(os.path.join(OUT_DIR, "blocks.jsonl")) if (b := json.loads(line))["prompt"]]
    clusters = {c["cluster"]: c for c in json.load(open(os.path.join(OUT_DIR, "clusters.json")))}
    # Prompts a coder labelled; those left blank once the outcome was settled were never judged.
    screened = {r["block_id"] for f in glob.glob(os.path.join(OUT_DIR, "screening", "round_*.csv"))
                for r in csv.DictReader(open(f, encoding="utf-8")) if r["fits"].strip()}
    return blocks, clusters, screened


def one_per_paper(blocks, n, rng):
    """Up to n of the blocks, in random order, at most one per paper."""
    pool = blocks[:]
    rng.shuffle(pool)
    out, seen = [], set()
    for b in pool:
        if b["paper_id"] not in seen:
            out.append(b)
            seen.add(b["paper_id"])
        if len(out) == n:
            break
    return out


def write_rows(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def screen():
    blocks, clusters, screened = load_clustered()
    rounds = glob.glob(os.path.join(OUT_DIR, "screening", "round_*.csv"))
    rng = random.Random(SEED + len(rounds) + 1)
    results = screening_results()
    rows, waiting = [], []
    for c, info in sorted(clusters.items()):
        if c < 0:
            continue
        # Screened with these members already: accepted, or waiting for an edit. A cluster
        # left open gets a fresh sample.
        if results.get(info["members"], "open") != "open":
            if results[info["members"]] != "accept":
                waiting.append(f"{c} ({results[info['members']]})")
            continue
        pool = [b for b in blocks if b["cluster"] == c and b["block_id"] not in screened]
        sample = one_per_paper(pool, paper_numbers.SCREEN_FIRST + paper_numbers.SCREEN_SECOND + SCREEN_RESERVE, rng)
        for i, b in enumerate(sample):
            rows.append({"cluster": c, "members": info["members"], "order": i + 1,
                         "terms": ", ".join(info["terms"][:6]), "block_id": b["block_id"],
                         "paper_id": b["paper_id"], "text": b["text"][:SCREEN_TEXT_CHARS], "fits": ""})
    if waiting:
        print("not re-screened, unchanged since their last round: " + ", ".join(waiting))
    if not rows:
        print("no cluster needs screening")
        return
    path = os.path.join(OUT_DIR, "screening", f"round_{len(rounds) + 1:02d}.csv")
    write_rows(path, rows, ["cluster", "members", "order", "terms", "block_id", "paper_id", "text", "fits"])
    print(f"{path}: {len(rows)} prompts from {len({r['cluster'] for r in rows})} clusters")


def import_screening(path, export_dir):
    """Fill `fits` in a screening round from the annotation page's exported labels.

    export_dir holds one JSON file per label ({"block_id", "fits"}), as the page's
    label store exports them; labels cleared on the page are cleared here too. The
    round's non-prompts replace the rows it added to BLOCK_REVIEW before.
    """
    fits = {}
    for f in glob.glob(os.path.join(export_dir, "*.json")):
        d = json.load(open(f))
        d = d.get("data", d)
        fits[d["block_id"]] = str(d["fits"])
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for r in rows:
        r["fits"] = fits.get(r["block_id"], "")
    write_rows(path, rows, list(rows[0]))

    source = os.path.basename(path)
    review = list(csv.DictReader(open(BLOCK_REVIEW, encoding="utf-8"))) if os.path.exists(BLOCK_REVIEW) else []
    review = [r for r in review if r["source"] != source]
    review += [{"block_id": r["block_id"], "keep": 0, "source": source} for r in rows if r["fits"] == "x"]
    write_rows(BLOCK_REVIEW, review, ["block_id", "keep", "source"])

    groups = {}
    for r in rows:
        groups.setdefault(r["cluster"], []).append(r)
    outcomes = [screen_outcome(g) for g in groups.values()]
    print(f"{path}: {sum(1 for r in rows if r['fits'])} labels, {sum(1 for r in rows if r['fits'] == 'x')} not prompts; "
          + ", ".join(f"{outcomes.count(o)} {o}" for o in ("accept", "revise", "open")))


def import_names(export_dir):
    """Record the names a coder set on the annotation page ({"cluster", "members", "name"})."""
    names, set_by_coder = cluster_names(), 0
    for f in glob.glob(os.path.join(export_dir, "*.json")):
        d = json.load(open(f))
        d = d.get("data", d)
        if d["name"].strip() and names.get(d["members"], {}).get("name") != d["name"].strip():
            names[d["members"]] = {"cluster": d["cluster"], "name": d["name"].strip()}
            set_by_coder += 1
    json.dump(names, open(NAMES, "w"), indent=1, ensure_ascii=False)
    print(f"{NAMES}: {set_by_coder} names set by a coder, {len(names)} in all")


def import_validation(labels_dir, resolved_dir):
    """Both coders' labels and the resolved label per prompt of the validation sample."""
    read = lambda d: [x.get("data", x) for x in (json.load(open(f)) for f in glob.glob(os.path.join(d, "*.json")))]
    labels, resolved = read(labels_dir), {r["prompt_id"]: r for r in read(resolved_dir)}
    first = {}
    for lab in labels:
        first[lab["coder"]] = min(first.get(lab["coder"], lab["at"]), lab["at"])
    coders = sorted(first, key=first.get)
    if len(coders) != 2:
        sys.exit(f"expected labels from two coders, found {len(coders)}")
    by = {(lab["coder"], lab["prompt_id"]): lab for lab in labels}
    rows = list(csv.DictReader(open(os.path.join(OUT_DIR, "validation_sample.csv"), encoding="utf-8")))
    for r in rows:
        a, b = (by.get((c, r["prompt_id"]), {}) for c in coders)
        r["fit_a"], r["fit_b"] = str(a.get("fit", "")), str(b.get("fit", ""))
        # type: the review group a coder picked as a better fit (group 0 is a real id).
        r["type_a"], r["type_b"] = ("" if x.get("type") is None else str(x["type"]) for x in (a, b))
        if r["fit_a"] and r["fit_a"] == r["fit_b"]:
            r["fit"] = r["fit_a"]
        else:
            r["fit"] = str(resolved.get(r["prompt_id"], {}).get("fit", ""))
        del r["text"]
    path = os.path.join(OUT_DIR, "validation.csv")
    write_rows(path, rows, ["prompt_id", "cluster", "name", "fit_a", "fit_b", "fit", "type_a", "type_b"])
    open_rows = sum(not (r["fit_a"] and r["fit_b"] and r["fit"]) for r in rows)
    print(f"{path}: {len(rows)} prompts, {sum(r['fit_a'] != r['fit_b'] for r in rows)} disagreements, {open_rows} not finished")


def review_groups():
    """The groups the validation labels, in TASK_TYPES order; prompts of REVIEW_NONE_GROUP
    are reviewed with the unassigned ones, as fitting none of them."""
    return [g["group"] for g in json.load(open(TASK_TYPES))["groups"] if g["group"] != REVIEW_NONE_GROUP]


def validation():
    """The sample is stratified by review group (review_groups); its `cluster` column holds
    the group's index (-1 for unassigned prompts and those of REVIEW_NONE_GROUP) and `name`
    the group's name."""
    blocks, _, screened = load_clustered()
    types = task_types()
    groups = review_groups()
    n = paper_numbers.sample_size(len(blocks), paper_numbers.CLUSTER_MARGIN)
    strata, names = {}, {-1: "(none of the groups)"}
    for b in blocks:
        g = types[b["cluster"]]["group"] if b["cluster"] >= 0 else None
        t = groups.index(g) if g in groups else -1
        strata.setdefault(t, []).append(b)
        if t >= 0:
            names[t] = g
    # Proportional allocation, rounded by largest remainder, at least one per stratum.
    share = {c: n * len(bs) / len(blocks) for c, bs in strata.items()}
    alloc = {c: max(1, int(s)) for c, s in share.items()}
    for c in sorted(share, key=lambda c: int(share[c]) - share[c])[:max(0, n - sum(alloc.values()))]:
        alloc[c] += 1
    rng = random.Random(SEED)
    rows = []
    for c, bs in sorted(strata.items()):
        pool = [b for b in bs if b["block_id"] not in screened]
        for b in one_per_paper(pool, alloc[c], rng):
            rows.append({"prompt_id": b["block_id"], "cluster": c, "name": names[c],
                         "text": b["text"][:SCREEN_TEXT_CHARS], "fit_a": "", "fit_b": "", "fit": ""})
    rng.shuffle(rows)
    path = os.path.join(OUT_DIR, "validation_sample.csv")
    write_rows(path, rows, ["prompt_id", "cluster", "name", "text", "fit_a", "fit_b", "fit"])
    print(f"{path}: {len(rows)} prompts (target {n}) from {len(strata)} strata")


if __name__ == "__main__":
    if sys.argv[1] == "import-screening":
        import_screening(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "import-validation":
        import_validation(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "import-names":
        import_names(sys.argv[2])
    else:
        {"cluster": cluster, "assign": assign, "second-pass": second_pass, "screen": screen,
         "validation": validation}[sys.argv[1]]()
