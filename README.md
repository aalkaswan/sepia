# SEPIA: A Dataset of Prompts from Software Engineering Research Papers

This package releases 8,799 LLM prompts mined from the LaTeX sources of
2,775 cs.SE papers on arXiv, together with their provenance and the code.

## Contents

| Path | What it holds |
|---|---|
| `prompts.jsonl` | one record per prompt (see below) |
| `papers.csv` | one record per paper with prompts: arXiv metadata and the version mined |
| `searched_papers.csv` | every searched paper with its number of prompts (the denominators) |
| `task_types.csv` | SE activity → task type → cluster, with prompt and paper counts |
| `prompt_embeddings_part1.npy`, `prompt_embeddings_part2.npy`, `prompt_embeddings_ids.csv` | the Harrier-OSS-v1-0.6B embedding of every prompt, as clustered (see "Explorer" below) |
| `annotations/` | all human labels (gold standard, screening, validation) |
| `code/` | the pipeline, its container, its configuration, and `reextract.py` |
| `explorer/` | the SEPIA explorer: an interactive cluster map and prompt search (see "Explorer" below) |

## Prompt records (`prompts.jsonl`)

| Field | Content |
|---|---|
| `prompt_id` | `<arXiv id>v<version>#<position>`, e.g. `2401.01234v2#3`; stable across releases |
| `arxiv_id`, `version`, `position` | the parts of the identifier; `position` counts the prompts of the paper from 0 in source order |
| `start`, `end` | character offsets of the prompt in the paper's assembled, comment-stripped TeX |
| `detector_patterns` | the detector patterns that found the prompt |
| `text` | the prompt as written in the LaTeX source, placeholders and markup kept; `null` when the paper's licence does not allow redistribution |
| `text_normalised` | the text with LaTeX stripped and placeholders masked as `[SLOT]`, as clustered; `null` like `text` |
| `text_sha256` | SHA-256 of `text` (UTF-8), always present, used to verify re-extracted text |
| `words` | number of words of the normalised text |
| `license` | the paper's arXiv licence, which also governs the prompt's text |
| `text_released` | whether `text` is included |
| `task_group` | the SE activity (Hou et al.'s six activities, LLM-centric tasks, Other), `null` if unassigned |
| `task_type` | one of the 16 task types, `null` if unassigned |
| `cluster_id`, `cluster` | the finer cluster and its name, `null` if unassigned |
| `second_pass` | whether the cluster comes from the second clustering pass over unassigned prompts |
| `family_id` | prompts whose embeddings have cosine similarity ≥ 0.95 share a family; deduplicate on it |
| `family_papers` | number of papers the family appears in (109 families appear in more than one) |

A prompt is a block the detector found and that was not later marked as not being a prompt.
A block may start with a few words of the sentence that introduced it.

The 11 patterns `detector_patterns` names are in `code/collect_prompts.py` (`PATTERNS`).
`prompt_env`, the one iteration 1 started from, matches any environment whose name contains
"prompt":

```python
("prompt_env", re.compile(
    r"\\begin\{(?P<env>\w*[Pp]rompt\w*)\}.{0,8000}?\\end\{(?P=env)\}", re.DOTALL), 0),
```

so `\begin{promptbox} ... \end{promptbox}` is a hit, and the `(?P=env)` backreference keeps the
match from closing on a different environment.

## Paper records (`papers.csv`)

`arxiv_id`, `version`, `title`, `authors`, `first_version_date`, `primary_category`,
`categories`, `license`, `doi`, `journal_ref` come from arXiv's metadata. `version` is the
version whose source we mined: for papers fetched from arXiv after the snapshot it is exact
(`version_source = fetched`, with the archive's `source_sha256`); for papers from the arXiv
snapshot it is the newest version the snapshot reports (`version_source = snapshot`), and
`version_proven` says whether the snapshot proves that the source belongs to it (it never does,
see "The arXiv snapshot" below). `prompts` counts the paper's prompts and `text_released` whether
their text is included.
`searched_papers.csv` lists every searched paper, with or without prompts.

## The arXiv snapshot

Paper metadata (titles, authors, categories, versions, licences) comes from arXiv's Open Archives
Initiative Protocol for Metadata Harvesting (OAI-PMH), the interface arXiv provides for bulk
metadata access. The LaTeX sources are not downloaded from arXiv one by one; they are read from an
existing dataset of arXiv sources. This keeps load off arXiv's servers, which ask that bulk
access not go through the regular site. Only the papers missing from that dataset were fetched
from arXiv directly (`fetch_arxiv_paper.py`), observing the 15-second crawl delay in arXiv's
`robots.txt`.

Papers were searched in a fixed snapshot of arXiv rather than on the live site, so the search is
repeatable. The snapshot is the arXiv Complete Corpus,
[`secemp9/arxiv-complete`](https://huggingface.co/datasets/secemp9/arxiv-complete) on Hugging Face:
3,148,796 papers, metadata harvested 2026-08-30 over OAI-PMH (`arXivRaw`), file mirror of
2026-09-05, submissions up to 2026-08-27. We do not redistribute it, since it carries the papers'
full texts. The pipeline expects its Parquet configs under `code/data/arxiv-complete/`: `metadata`
(`paper_id`, `title`, `authors`, `categories`, `first_version_date`, `latest_version_date`,
`arxiv_abs_url`), `versions` (`paper_id`, `version`, `version_date`, `has_pdf`, `has_ps`,
`has_html`, `reported_size_kb`, `is_metadata_only`), and `paper_text`, which holds one assembled
TeX text per paper. `collect_prompts.py` runs the detector over the cs subset drawn from
`paper_text`, `datasets/cs_arxiv_latex_dataset` (`id`, `title`, `categories`, `latex`), and
`fetch_arxiv_paper.py` queries `metadata` and `versions` for a single paper. The corpus itself is
a CC0 compilation; each paper keeps its own arXiv licence, which is what governs the prompt text.

`code/arxiv_papers.csv` lists every arXiv paper with cs.SE among its categories submitted between
2023-01-01 and 2026-10-31 (18,339 papers, 13,836 of them with cs.SE as the primary category),
which reaches two months past the snapshot's last submission:

| Papers | Set |
|---:|---|
| 18,339 | listed (cs.SE in `categories`, 2023-01-01 to 2026-10-31) |
| 14,775 | LaTeX in the snapshot |
| 3,564 | missing from the snapshot, mostly submitted after it was taken |
| 2,189 | of those, e-print fetched from arXiv afterwards with TeX in it |
| 1,375 | no TeX anywhere (PDF-only submission, or not found): not searched |
| 16,964 | searched, i.e. listed with LaTeX (`searched_papers.csv`) |

The snapshot's source packages are unversioned: it reports a paper's versions in `versions`, but
its LaTeX is not tied to one. For the 14,775 snapshot papers, `papers.csv` therefore records the
newest version the snapshot reports (`version_source = snapshot`) and `version_proven = 0`
throughout this release. The fetched papers name their version in the e-print URL and carry the
archive's SHA-256 in `source_sha256` (`version_source = fetched`, `version_proven = 1`). This is
why `reextract.py` falls back to earlier versions and verifies every text against `text_sha256`.

`start` and `end` point into one assembled, comment-stripped text per paper, built the way the
snapshot's own builder built `paper_text` (its README's "Assembly and omitted files"):

- a single `.tex`/`.ltx` file is taken as is;
- otherwise the file with `\documentclass` is expanded depth-first through `\input`, `\include`
  and `\subfile`, keeping the longest result;
- when no main file is found, or the expansion covers less than half the TeX bytes, every TeX file
  is concatenated in path order;
- a formatted `.bbl` is appended when its text is not already in the result;
- whole-line `%` comments are then dropped, so commented-out figures and listings are not searched.

`build_added_dataset.py` applies the same rules to the fetched e-prints, and `reextract.py` reuses
that code. The snapshot's builder resolved some `\input` commands differently, so `reextract.py`
also scans each TeX file on its own and matches blocks by digest; that is why a prompt can come
back as `found` rather than `matched`.

## Licensing

arXiv papers carry different licences, and arXiv's default licence grants no redistribution
rights to third parties. We therefore include the prompt text only for papers under a Creative
Commons licence: 5,878 prompts (67%). Each prompt's text remains under
its paper's licence, given in `license`; mind NonCommercial and NoDerivatives terms when you
reuse it. The other 2,921 prompts are pointers; `code/reextract.py` re-creates
their text from the arXiv source:

```
python code/reextract.py --download --email you@example.org   # downloads e-prints politely
```

It writes `prompts_full.jsonl` and marks each prompt `matched` (same version and position),
`found` (elsewhere in that version's files), `found in vN` (an earlier version), `not found` or
`no source`. Only the standard library is needed. The arXiv snapshot's sources are
unversioned, so the script falls back to earlier versions until every digest matches; the
digest guards against wrong text. Downloads are cached in `eprints/`, so rerunning retries
only the papers whose download failed.

arXiv metadata is CC0. Our annotations, task types and other data are released under
CC BY 4.0, and the code under the Apache License 2.0. Full texts and PDFs are not included.
The prompt embeddings (`prompt_embeddings_*.npy`) cover all 8,799 prompts, including the 2,921
whose text is not released; they are vectors computed from the text and do not contain it.

## Annotations (`annotations/`)

Coders are anonymised as A and B. Quoted prompt text (`first_words`) is kept only for papers
whose licence allows it.

| File | Content |
|---|---|
| `gold/sample.csv` | the gold-standard papers and whether the detector flagged them |
| `gold/annotator_A.csv`, `gold/annotator_B.csv` | each annotator's independent prompt list per paper |
| `gold/adjudicated.csv` | the adjudicated has-prompts label where the annotators disagreed |
| `gold/prompts.csv` | the reconciled gold prompts: who found them, where (`main`, `appendix`, `figure`, `listing`) and in what form (`text`, `image`) |
| `detector_development.csv` | the per-round prompt / no-prompt labels used to develop the detector |
| `not_prompts.csv` | blocks found not to be prompts (`keep = 0`), and the review round |
| `cluster_screening.csv` | the cluster-screening labels: `1` fits the cluster, `0` misfit, `x` not a prompt |
| `cluster_validation.csv` | the two-coder validation of 96 prompts against the SE activities, with the resolved label `fit` and the activity a coder found better |

`block_id` and `prompt_id` use the same identifier scheme; blocks that are not prompts are not
in `prompts.jsonl`. The task types themselves, and the clusters each covers, are in
`code/data/clusters/task_types.json`, as the pipeline reads them.


## Reproducing the pipeline

The code runs in one container (`code/docker/cluster.Dockerfile`); nothing else needs installing.

1. `build_added_dataset.py` assembles the e-prints of the papers the snapshot is missing
   ("The arXiv snapshot" above); `code/arxiv_papers.csv` lists every paper the search listed.
2. `collect_prompts.py` runs the detector (`--hits-only` for the frozen detector).
3. `gold_sample.py`, `gold_reconcile.py` and `gold_agreement.py` draw and score the gold standard.
4. `task_clusters.py cluster | screen | assign | second-pass | validation` builds the task types,
   with the edits in `data/clusters/`.
5. `cluster_results.py` and `paper_numbers.py` write the figures, tables and every number of
   the paper.

Rebuilding the clusters needs every prompt's text: run `reextract.py` first.

## Explorer (`explorer/`)

The SEPIA explorer is a small web app over the dataset: a cluster map of every prompt (a 2D UMAP
projection of its embedding), semantic and keyword prompt search, nearest neighbours of a prompt,
and suggested task types for unassigned prompts. Semantic search embeds the query exactly as the
pipeline embedded the prompts (Harrier-OSS-v1-0.6B at the pinned revision, same instruction), and,
like the release, the app serves prompt text only where `text_released` is true. It is also what
the Hugging Face Space runs (`explorer/Dockerfile`); `explorer/README.md` describes how to run it.

Its data bundle (`explorer/data/`) is built from the release:

```
pip install umap-learn
python explorer/build_data.py --prompts prompts.jsonl --papers papers.csv \
    --emb prompt_embeddings_part1.npy prompt_embeddings_part2.npy --ids prompt_embeddings_ids.csv
```

The embeddings are float32, 1,024 dimensions, normalised to unit length, split in two parts of
4,400 and 4,399 rows. Row `row` of `prompt_embeddings_ids.csv` gives the `prompt_id` of each
embedding, in the order of `prompts.jsonl`. They are the vectors `task_clusters.py` computed and
clustered (the rows of `code/data/clusters/embeddings.npy` and `added_embeddings.npy` that belong
to prompts), so they cover every prompt, including those whose text is not released.
