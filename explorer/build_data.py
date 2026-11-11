"""Build the data bundle for the SEPIA explorer.

Inputs: the released prompts.jsonl and papers.csv, plus the prompt embeddings
(two .npy parts and the row -> prompt_id CSV). Outputs, in ./data:
  points.json      what the browser needs to draw the map (coordinates, labels, snippets)
  records.json     per-prompt details the server returns on request (full released text)
  embeddings.npy   the embeddings as float16, in the same row order as records.json

Usage:
  python build_data.py --prompts prompts.jsonl --papers papers.csv \
      --emb part1.npy part2.npy --ids prompt_embeddings_ids.csv
"""
import argparse
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")

# Order of the activity groups as in the paper's task-type table.
GROUP_ORDER = [
    "Requirements engineering", "Software design", "Software development",
    "Software quality assurance", "Software maintenance", "Software management",
    "LLM-centric tasks", "Other", "Unassigned",
]
SNIPPET = 220


def none(v):
    return None if v in (None, "None", "", "nan") else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--papers", required=True)
    ap.add_argument("--emb", nargs="+", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--neighbors", type=int, default=30)
    ap.add_argument("--min-dist", type=float, default=0.12)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    prompts = [json.loads(line) for line in open(args.prompts, encoding="utf-8")]
    with open(args.ids, newline="") as f:
        ids = [r["prompt_id"] for r in csv.DictReader(f)]
    emb = np.concatenate([np.load(p, allow_pickle=False) for p in args.emb]).astype(np.float32)
    assert emb.shape[0] == len(ids) == len(prompts), (emb.shape, len(ids), len(prompts))
    by_id = {p["prompt_id"]: p for p in prompts}
    prompts = [by_id[i] for i in ids]  # embedding row order
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    papers = {}
    with open(args.papers, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            papers[r["arxiv_id"]] = r

    # 2D map. UMAP on cosine distance, as in the clustering pipeline.
    import umap
    xy = umap.UMAP(n_neighbors=args.neighbors, min_dist=args.min_dist, metric="cosine",
                   random_state=42, n_components=2).fit_transform(emb)
    xy = (xy - xy.mean(0)) / xy.std(0)

    types, clusters, paper_list = [], [], []
    type_idx, cluster_idx, paper_idx = {}, {}, {}
    type_group = []
    cols = {k: [] for k in ("x", "y", "g", "t", "c", "p", "r", "f", "w")}
    snippets, records = [], []
    for i, p in enumerate(prompts):
        group = none(p["task_group"]) or "Unassigned"
        ttype = none(p["task_type"]) or "Unassigned"
        cname = none(p["cluster"]) or "Unassigned"
        if ttype not in type_idx:
            type_idx[ttype] = len(types)
            types.append(ttype)
            type_group.append(GROUP_ORDER.index(group))
        if cname not in cluster_idx:
            cluster_idx[cname] = len(clusters)
            clusters.append(cname)
        aid = p["arxiv_id"]
        if aid not in paper_idx:
            pr = papers.get(aid, {})
            paper_idx[aid] = len(paper_list)
            paper_list.append({
                "id": aid,
                "v": pr.get("version") or p["version"],
                "title": (pr.get("title") or "").strip(),
                "year": int((pr.get("first_version_date") or "0000")[:4] or 0),
                "authors": (pr.get("authors") or "").split("; ")[:3],
                "nauthors": len((pr.get("authors") or "").split("; ")) if pr.get("authors") else 0,
            })
        released = str(p["text_released"]) == "True"
        text = p["text"] if released else None
        cols["x"].append(round(float(xy[i, 0]), 4))
        cols["y"].append(round(float(xy[i, 1]), 4))
        cols["g"].append(GROUP_ORDER.index(group))
        cols["t"].append(type_idx[ttype])
        cols["c"].append(cluster_idx[cname])
        cols["p"].append(paper_idx[aid])
        cols["r"].append(1 if released else 0)
        cols["f"].append(int(none(p["family_papers"]) or 1))
        cols["w"].append(int(none(p["words"]) or 0))
        snip = None
        if released:
            s = " ".join((p.get("text_normalised") or text or "").split())
            snip = s[:SNIPPET] + ("…" if len(s) > SNIPPET else "")
        snippets.append(snip)
        records.append({
            "prompt_id": p["prompt_id"],
            "text": text,
            "text_normalised": p.get("text_normalised") if released else None,
            "license": p.get("license"),
            "family_id": none(p.get("family_id")),
            "second_pass": str(p.get("second_pass")) == "True",
            "patterns": p.get("detector_patterns"),
        })

    points = {
        "groups": GROUP_ORDER,
        "types": types,
        "typeGroup": type_group,
        "clusters": clusters,
        "papers": paper_list,
        "cols": cols,
        "snippet": snippets,
        "ids": [p["prompt_id"] for p in prompts],
    }
    with open(os.path.join(OUT, "points.json"), "w", encoding="utf-8") as f:
        json.dump(points, f, ensure_ascii=False, separators=(",", ":"))
    with open(os.path.join(OUT, "records.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, separators=(",", ":"))
    np.save(os.path.join(OUT, "embeddings.npy"), emb.astype(np.float16))
    print(f"{len(prompts)} prompts, {len(types)} types, {len(clusters)} clusters, {len(paper_list)} papers")


if __name__ == "__main__":
    main()
