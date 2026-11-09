"""Results of the task clustering: figures, the task-type table, and the counts behind them.

Reads data/clusters/ (blocks.jsonl, clusters.json) as task_clusters.py left it, with
the clusters grouped into task types by data/clusters/task_types.json, and writes:
  paper/figures/task_types.pdf   papers per task type
  paper/figures/task_years.pdf   share of each year's papers that use each SE activity (group of types)
  paper/generated/task_table.tex the task types with their counts
  data/clusters/results.json     the counts, read by paper_numbers.py

A paper counts for a type when at least one of its prompts is in that type's cluster;
a paper's year is its arXiv submission year. Runs in the clustering container:
  docker run --rm -u $(id -u):$(id -g) -v "$PWD":/work -w /work se-prompts-cluster \\
      python cluster_results.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLUSTER_DIR = "data/clusters"
FIG_DIR = "paper/figures"
GEN_DIR = "paper/generated"
UNASSIGNED = "Unassigned"
# Reference palette (dataviz skill): one blue for magnitude, grey for the unassigned prompts.
BLUE, GREY, INK, MUTED = "#2a78d6", "#a9a8a2", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
                     "pdf.fonttype": 42})
COLUMN_WIDTH = 3.5  # inches, the ACM two-column format


def year(paper_id):
    """arXiv identifiers start with YYMM."""
    return 2000 + int(paper_id[:2])


def load():
    """Prompt blocks with their task type (`type`, -1 for unassigned), and the types in
    task_types.json order with the clusters each covers."""
    from task_clusters import task_types
    blocks = [b for line in open(os.path.join(CLUSTER_DIR, "blocks.jsonl")) if (b := json.loads(line))["prompt"]]
    of_cluster = task_types()
    for b in blocks:
        b["type"] = of_cluster[b["cluster"]]["type"] if b["cluster"] >= 0 else -1
    spec = json.load(open(os.path.join(CLUSTER_DIR, "task_types.json")))
    types = [{"cluster": i, "name": t["name"], "group": g["group"], "parts": t["clusters"]}
             for i, (g, t) in enumerate((g, t) for g in spec["groups"] for t in g["types"])]
    return blocks, types


def counts(blocks, task_types):
    """Per task type (`cluster` is the type id, -1 for unassigned): prompts, papers and
    each year's share of papers."""
    papers_all = {b["paper_id"] for b in blocks}
    years = sorted({year(p) for p in papers_all})
    per_year = {y: len({p for p in papers_all if year(p) == y}) for y in years}
    types = []
    for c in task_types + [{"cluster": -1, "name": UNASSIGNED, "group": None, "parts": []}]:
        bs = [b for b in blocks if b["type"] == c["cluster"]]
        papers = {b["paper_id"] for b in bs}
        types.append({
            "cluster": c["cluster"], "name": c["name"], "group": c["group"], "parts": c["parts"],
            "prompts": len(bs), "papers": len(papers), "families": len({b["family"] for b in bs}),
            "paper_share": len(papers) / len(papers_all),
            "by_year": {y: len({p for p in papers if year(p) == y}) / per_year[y] for y in years},
        })
    named = sorted((t for t in types if t["cluster"] >= 0), key=lambda t: -t["papers"])
    # The finer clusters (main and second pass) by name, for trends within a type.
    names = {c["cluster"]: c["name"] for c in json.load(open(os.path.join(CLUSTER_DIR, "clusters.json")))}
    cluster_papers = {}
    for b in blocks:
        if b["cluster"] >= 0:
            cluster_papers.setdefault(names[b["cluster"]], set()).add(b["paper_id"])
    clusters = [{"name": name, "papers": len(ps),
                 "by_year": {y: len({p for p in ps if year(p) == y}) / per_year[y] for y in years}}
                for name, ps in sorted(cluster_papers.items(), key=lambda x: -len(x[1]))]
    # Papers whose prompts span several task types.
    types_of = {}
    for b in blocks:
        if b["type"] >= 0:
            types_of.setdefault(b["paper_id"], set()).add(b["type"])
    # The SE activities (groups of task types), in task_types.json order.
    groups = {t["group"]: set() for t in task_types}
    group_of = {t["cluster"]: t["group"] for t in task_types}
    for b in blocks:
        if b["type"] >= 0:
            groups[group_of[b["type"]]].add(b["paper_id"])
    return {"prompts": len(blocks), "papers": len(papers_all), "years": years, "papers_per_year": per_year,
            "types": named + [t for t in types if t["cluster"] < 0], "clusters": clusters,
            "groups": [{"name": g, "papers": len(ps), "paper_share": len(ps) / len(papers_all),
                        "by_year": {y: len({p for p in ps if year(p) == y}) / per_year[y] for y in years}}
                       for g, ps in groups.items()],
            "typed_papers": len(types_of), "multi_type_papers": sum(len(t) > 1 for t in types_of.values())}


def plot_types(res):
    """Horizontal bars: papers with at least one prompt of each type (the unassigned prompts are left out)."""
    types = [t for t in res["types"] if t["cluster"] >= 0]
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 0.135 * len(types) + 0.4))
    ys = range(len(types))[::-1]
    for y, t in zip(ys, types):
        ax.barh(y, t["papers"], height=0.72, color=BLUE)
        ax.text(t["papers"] + 3, y, f"{t['papers']}", va="center", ha="left", color=MUTED, fontsize=6)
    ax.set_yticks(list(ys), [t["name"] for t in types])
    ax.set_xlabel("Papers")
    ax.set_xlim(0, max(t["papers"] for t in types) * 1.12)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(FIG_DIR, "task_types.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "task_types.png"), dpi=200)


def plot_years(res):
    """Heatmap: share of each year's papers with at least one prompt of the SE activity
    (single-hue scale), most papers first."""
    types = sorted(res["groups"], key=lambda g: -g["papers"])
    years = res["years"]
    grid = [[t["by_year"][y] for y in years] for t in types]
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 0.135 * len(types) + 0.55))
    ax.imshow(grid, cmap="Blues", vmin=0, vmax=max(max(r) for r in grid), aspect="auto")
    top = max(max(r) for r in grid)
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=5.5,
                    color="white" if v > 0.55 * top else INK)
    ax.set_yticks(range(len(types)), [t["name"] for t in types])
    ax.set_xticks(range(len(years)), [f"{y}\n({res['papers_per_year'][y]})" for y in years])
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=2)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(FIG_DIR, "task_years.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "task_years.png"), dpi=200)


def tex_escape(s):
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def write_table(res):
    """Task types under their SE activity (task_types.json order), with prompts and
    papers; unassigned prompts last."""
    n = lambda x: f"{x:,}".replace(",", "{,}")
    rows, group = [], None
    for t in sorted(res["types"], key=lambda t: (t["cluster"] < 0, t["cluster"])):
        if t["group"] != group and t["group"] is not None:
            rule = r"\midrule" + "\n" if group is not None else ""
            rows.append(rule + rf"\multicolumn{{3}}{{@{{}}l}}{{\textbf{{{tex_escape(t['group'])}}}}} \\")
            group = t["group"]
        if t["cluster"] < 0:
            rows.append(r"\midrule")
        name = r"\emph{" + t["name"] + "}" if t["cluster"] < 0 else r"\quad " + tex_escape(t["name"])
        rows.append(f"{name} & {n(t['prompts'])} & {n(t['papers'])} \\\\")
    body = "\n".join(rows)
    tex = rf"""% Written by cluster_results.py; do not edit.
\begin{{table}}[t]
  \caption{{Task types of the prompts, grouped by SE activity. Papers counts the papers with at least one prompt of the type.}}
  \label{{tab:task-types}}
  \footnotesize
  \begin{{tabular}}{{@{{}}l r r@{{}}}}
    \toprule
    Task type & Prompts & Papers \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table}}
"""
    open(os.path.join(GEN_DIR, "task_table.tex"), "w").write(tex)


if __name__ == "__main__":
    blocks, types = load()
    res = counts(blocks, types)
    os.makedirs(FIG_DIR, exist_ok=True)
    plot_types(res)
    plot_years(res)
    write_table(res)
    json.dump(res, open(os.path.join(CLUSTER_DIR, "results.json"), "w"), indent=1)
    for t in res["types"]:
        print(f"{t['prompts']:5} {t['papers']:5} {100 * t['paper_share']:5.1f}%  {t['name']}")
    print(f"{res['prompts']} prompts in {res['papers']} papers; papers per year {res['papers_per_year']}")
