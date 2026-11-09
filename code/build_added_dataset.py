"""Build the LaTeX dataset of papers added after the snapshot (stage 1, late batch).

The snapshot (datasets/cs_arxiv_latex_dataset) lacked 3,564 listed papers, mostly
submitted after it was taken. Their arXiv e-prints were fetched afterwards into one
archive, data/missing_snapshot/missing_from_snapshot_new.zip: per paper a folder with
arxiv_source.tar.gz (and the PDF, which is not used), plus manifest.csv and
skipped.csv (papers with no TeX source: PDF-only submissions or not found).

Each paper's TeX is assembled the way the snapshot's paper_text was
(data/arxiv-complete/README.md, "Assembly and omitted files"): a single .tex/.ltx file
as is; otherwise the main file with \\input, \\include and \\subfile expanded
depth-first; all TeX files in path order when there is no main file or the expansion
covers less than half the TeX bytes; formatted .bbl files appended when not included.
Archives are read in memory and never extracted. Writes datasets/added_latex_dataset
(id, title, categories, latex, as the snapshot) and data/prompts/added/papers.csv
(paper, resolution, or the reason it has no TeX).

Runs in the clustering container (it has `datasets`):
  docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -v "$PWD":/work -w /work \\
      se-prompts-cluster python -I build_added_dataset.py
"""
import csv
import gzip
import io
import os
import posixpath
import re
import tarfile
import zipfile

ARCHIVE = "data/missing_snapshot/missing_from_snapshot_new.zip"
ROOT = "missing_from_snapshot_new"
OUT = "datasets/added_latex_dataset"
PAPERS = "data/prompts/added/papers.csv"
TEX = (".tex", ".ltx")
INCLUDE = re.compile(r"\\(?:input|include|subfile)\s*\{([^}]+)\}|\\input\s+([^\s\\%{}]+)")
MAIN = re.compile(r"\\documentclass")


def decode(data):
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def source_files(blob):
    """{path: text} of the TeX and .bbl files in an e-print (a tar.gz, or one gzipped file)."""
    try:
        tar = tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz")
    except tarfile.ReadError:
        try:
            data = gzip.decompress(blob)
        except OSError:
            data = blob
        return {} if data[:4] == b"%PDF" else {"main.tex": decode(data)}
    files = {}
    for m in tar.getmembers():
        name = posixpath.normpath(m.name.lstrip("./"))
        if m.isfile() and name.lower().endswith(TEX + (".bbl",)):
            files[name] = decode(tar.extractfile(m).read())
    return files


def commented(text, pos):
    """Whether pos is after an unescaped % on its line."""
    line = text[text.rfind("\n", 0, pos) + 1:pos]
    return re.search(r"(?<!\\)%", line) is not None


def expand(path, files, used):
    """The file's text with its includes expanded depth-first; records the files used."""
    used.add(path)
    text = files[path]
    out, last = [], 0
    for m in INCLUDE.finditer(text):
        if commented(text, m.start()):
            continue
        name = (m.group(1) or m.group(2)).strip()
        # Relative to the archive root, as TeX resolves them, or to the including file.
        candidates = [posixpath.normpath(p) for n in (name, name + ".tex")
                      for p in (n, posixpath.join(posixpath.dirname(path), n))]
        target = next((c for c in candidates if c in files), None)
        if target is None or target in used:
            continue
        out.append(text[last:m.start()])
        out.append(expand(target, files, used))
        last = m.end()
    out.append(text[last:])
    return "".join(out)


def assemble(files):
    """(latex, resolution) for one paper's files."""
    tex = {p: t for p, t in files.items() if p.lower().endswith(TEX)}
    bbl = [p for p in sorted(files) if p.lower().endswith(".bbl")]
    if not tex:
        return None, "no tex"
    if len(tex) == 1:
        text, used, resolution = next(iter(tex.values())), set(tex), "single"
    else:
        best = None
        for p in sorted(p for p, t in tex.items() if MAIN.search(t)):
            used = set()
            text = expand(p, tex, used)
            if best is None or len(text) > len(best[0]):
                best = (text, used)
        covered = sum(len(tex[p]) for p in best[1]) if best else 0
        if best and covered >= sum(len(t) for t in tex.values()) / 2:
            (text, used), resolution = best, "resolved"
        else:
            text, used, resolution = "\n".join(tex[p] for p in sorted(tex)), set(tex), "fallback"
    for p in bbl:
        if files[p] not in text:
            text += "\n" + files[p]
    return text, resolution


def main():
    from datasets import Dataset

    listed = {r["arxiv_id"]: r for r in csv.DictReader(open("arxiv_papers.csv", encoding="utf-8"))}
    rows, log = [], []
    with zipfile.ZipFile(ARCHIVE) as z:
        for r in csv.DictReader(io.TextIOWrapper(z.open(f"{ROOT}/skipped.csv"), encoding="utf-8")):
            log.append({"paper": r["paper"], "resolution": r["reason"]})
        sources = sorted(n for n in z.namelist()
                         if n.startswith(f"{ROOT}/") and n.endswith("/arxiv_source.tar.gz"))
        for name in sources:
            pid = name.split("/")[1]
            latex, resolution = assemble(source_files(z.read(name)))
            log.append({"paper": pid, "resolution": resolution})
            if latex is not None:
                rows.append({"id": pid, "title": listed[pid]["title"], "categories": listed[pid]["categories"],
                             "latex": latex})
    Dataset.from_list(rows).save_to_disk(OUT)
    os.makedirs(os.path.dirname(PAPERS), exist_ok=True)
    with open(PAPERS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["paper", "resolution"])
        w.writeheader()
        w.writerows(sorted(log, key=lambda r: r["paper"]))
    counts = {}
    for r in log:
        counts[r["resolution"]] = counts.get(r["resolution"], 0) + 1
    print(f"{len(rows)} papers with TeX -> {OUT}; {counts}")


if __name__ == "__main__":
    main()
