"""Iteratively build a prompt detector for arXiv LaTeX, one reviewed pattern at a time.

Each iteration runs the adopted PATTERNS over the dataset and samples papers for
manual review: 20 the detector flags (to check precision) and 20 papers it
does not flag (to look for misses). Papers sampled in an earlier
iteration are never sampled again. CANDIDATES holds patterns that are proposed
but not adopted yet; the review sheet shows which of them would fire on each
paper, so a candidate is only adopted once the review supports it.

Usage (in a container, like find_prompts.py):
    docker run --rm -e PYTHONDONTWRITEBYTECODE=1 -v "$PWD":/workspace -w /workspace \\
        python:3.12-slim sh -c "pip install -q datasets==3.6.0 && \\
        python3 collect_prompts.py --iteration 1 && \\
        chown -R $(id -u):$(id -g) data/prompts/iterations"

The container runs as root, so the trailing chown (with ids expanded by the
host shell) hands the output back to your user and keeps the review sheet editable.

Once the patterns are frozen, `--hits-only PATH` runs them on another dataset (papers
added later, e.g. --dataset datasets/added_latex_dataset --hits-only
data/prompts/added/hits.jsonl) without drawing a review sample.
"""
import argparse
import csv
import json
import os
import random
import re

# Patterns are (name, regex, trail[, lead]). A pattern that matches a whole
# block has trail 0; one that matches only a marker (a "Prompt:" label, a chat
# token) keeps `trail` characters after the match, and `lead` characters before
# it, so the hit shows the prompt itself.
INSTRUCTION_WORDING = (
    r"\b(?:you are (?:an|a)\b|your task is|you (?:will be|are) given"
    r"|please (?:answer|provide|generate|summarize|write|classify|explain))"
)


def inside(env, n):
    """Up to n characters that stay inside the current `env` block (named group), so a
    match never runs past the block's \\end into the next one."""
    return rf"(?:(?!\\end\{{(?P={env})\}}).){{0,{n}}}?"


BOXES = r"tcolorbox|tcblisting|mdframed|framed|shaded|AIbox"
# Box options, either [key=val,...] or tcblisting's {key={...},...}
BOX_OPTS = r"(?P<opts>\[[^\]]{0,800}\]|\{(?:[^{}]|\{[^{}]*\}){0,800}\})"
# A caption naming a prompt as the thing shown, e.g. "Prompt template for ...".
CAPTION_PROMPT = (r"\\caption\{\s*(?:(?:[\w\-]+\s+){0,3})?[Pp]rompts?\b"
                  r"|\\caption\{[^}]{0,120}?\bprompts?\s+(?:templates?|structures?|formats?)\b")

# Adopted patterns. Iteration 1 started with prompt_env alone.
PATTERNS = [
    # Any environment whose name contains "prompt", e.g. \begin{promptbox}...\end{promptbox}
    ("prompt_env", re.compile(
        r"\\begin\{(?P<env>\w*[Pp]rompt\w*)\}.{0,8000}?\\end\{(?P=env)\}", re.DOTALL), 0),
    # Adopted after iteration 1 review:
    # Code listings whose caption or title mentions "prompt" (narrowed in iteration 4:
    # language=prompt alone was a code style, not a prompt)
    ("listing_caption_prompt", re.compile(
        r"\\begin\{(?P<env>lstlisting|minted)\}\[[^\]]{0,400}?(?:caption|title)\s*=\s*\{?[^\]]{0,200}?[Pp]rompt[^\]]{0,400}\]"
        + inside("env", 8000) + r"\\end\{(?P=env)\}", re.DOTALL), 0),
    # Quote/verbatim/listing bodies with wording addressed to a model
    ("instruction_block", re.compile(
        r"\\begin\{(?P<env>quote|quotation|verbatim\*?|lstlisting)\}(?:\[[^\]]{0,400}\])?" + inside("env", 3000)
        + INSTRUCTION_WORDING + inside("env", 5000) + r"\\end\{(?P=env)\}", re.DOTALL | re.I), 0),
    # A "Prompt:" label at the start of a line, e.g. \textbf{Prompt:} or "Prompt 1:"
    ("prompt_label_line", re.compile(
        r"(?im)^\s*(?:\\(?:textbf|textit|emph)\{)?(?:system |user |task )?prompt(?:\s*\d+)?\}?\s*:"), 1500),
    # Chat-model special tokens; LaTeX sources sometimes write <<SYS>> as guillemets
    ("special_tokens", re.compile(r"\[INST\]|<<SYS>>|«SYS»|<\|(?:system|user|assistant|im_start)\|>"), 1500),
    # Adopted after iteration 2 review:
    # Boxes whose body is addressed to a model, whatever their title says
    ("instruction_box", re.compile(
        rf"\\begin\{{(?P<env>{BOXES})\}}{BOX_OPTS}?" + inside("env", 3000)
        + INSTRUCTION_WORDING + inside("env", 5000) + r"\\end\{(?P=env)\}", re.DOTALL | re.I), 0),
    # A figure or table whose caption names a prompt. Narrowed in iteration 4 to the
    # float itself, and only floats without \includegraphics: a prompt shown as an
    # image has no text to collect.
    ("prompt_caption", re.compile(
        r"\\begin\{(?P<env>figure\*?|table\*?)\}(?:(?!\\end\{(?P=env)\}|\\includegraphics).){0,8000}?"
        rf"(?:{CAPTION_PROMPT})(?:(?!\\end\{{(?P=env)\}}|\\includegraphics).){{0,3000}}?\\end\{{(?P=env)\}}",
        re.DOTALL), 0),
    # Adopted after iteration 3 review:
    # Boxes whose title/options mention "prompt"
    ("box_title_prompt", re.compile(
        rf"\\begin\{{(?P<env>{BOXES})\}}{BOX_OPTS}" + inside("env", 8000)
        + r"\\end\{(?P=env)\}", re.DOTALL), 0),
    # A table with a "Prompt" header cell, where each row holds a prompt
    ("prompt_table_column", re.compile(
        r"\\begin\{(?P<env>tabular\*?|tabularx|longtable)\}" + inside("env", 1500)
        + r"\\textbf\{Prompts?\}\s*(?:&|\\\\)" + inside("env", 8000) + r"\\end\{(?P=env)\}", re.DOTALL), 0),
    # Adopted after iteration 4 review:
    # A quote/verbatim/listing/box opened right after the word "prompt"
    ("block_after_prompt_mention", re.compile(
        rf"prompt[^\n]{{0,200}}\n?\s*\\begin\{{(?P<env>quote|quotation|verbatim\*?|lstlisting|{BOXES})\}}"
        + inside("env", 8000) + r"\\end\{(?P=env)\}", re.DOTALL | re.I), 0),
    # Adopted after iteration 5 review:
    # A listing or text-only float whose \label names a prompt, e.g. label={lst:prompt_wntr}, \label{fig:prompt}
    ("label_prompt", re.compile(
        r"\\begin\{(?P<env>lstlisting|minted|figure\*?|table\*?)\}(?:(?!\\end\{(?P=env)\}|\\includegraphics).){0,8000}?"
        r"label\s*=?\s*\{[^}]*prompt[^}]*\}(?:(?!\\end\{(?P=env)\}|\\includegraphics).){0,3000}?\\end\{(?P=env)\}",
        re.DOTALL | re.I), 0),
]

# Extra conditions on a pattern's match, checked after the regex.
CAPTION_OR_LABEL = re.compile(r"\\(?:caption|label)\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}")
WORD = re.compile(r"\b[A-Za-z]{3,}\b")
NUMBER = re.compile(r"\d")
COMMAND_NAME = re.compile(r"\\[A-Za-z]+\*?")


def is_wordy(block, min_words=25):
    """True for a float or table that reads like text rather than results: outside
    its caption and label, at least `min_words` words and three times as many words
    as numbers."""
    body = COMMAND_NAME.sub(" ", CAPTION_OR_LABEL.sub(" ", block))
    words = len(WORD.findall(body))
    numbers = len(re.findall(r"\d+(?:[.,]\d+)?", body))
    return words >= min_words and words >= 3 * numbers


REQUIRE = {
    # Results tables also have prompt captions and "Prompt" columns.
    "prompt_caption": lambda m: is_wordy(m.group(0)),
    "prompt_table_column": lambda m: is_wordy(m.group(0)),
    "label_prompt": lambda m: is_wordy(m.group(0)),
    "box_title_prompt": lambda m: re.search(r"[Pp]rompt", m.group("opts")),
}

# Proposed patterns, not used for flagging. Shown in the review sheet only.
CANDIDATES = [
    # New for iteration 4:
    # A quote/verbatim/box that opens with a request to a model, e.g. "Write a ...", "Given the following ..."
    ("imperative_block", re.compile(
        rf"\\begin\{{(?P<env>quote|quotation|verbatim\*?|lstlisting|{BOXES})\}}(?:\[[^\]]{{0,400}}\])?\s*"
        r"(?:\\\w+(?:\{[^}]*\})?\s*)*(?:Write|Generate|Given|Create|Implement|Translate|Summari[sz]e|Classify"
        r"|Act as|Rewrite|Review|Identify|Determine|Analy[sz]e|I want)\b" + inside("env", 8000) + r"\\end\{(?P=env)\}",
        re.DOTALL), 0),
    # Chat transcript lines, e.g. "System: ..." / "User: ..." / "Assistant: ..."
    ("chat_role_lines", re.compile(
        r"(?m)^\s*(?:\\textbf\{)?(?:System|User|Human)\}?\s*:.{0,1500}?^\s*(?:\\textbf\{)?(?:Assistant|AI|Model|User)\}?\s*:",
        re.DOTALL), 1200),
    # More wording addressed to a model, in a quote/verbatim/listing/box
    ("more_instruction_wording", re.compile(
        rf"\\begin\{{(?P<env>quote|quotation|verbatim\*?|lstlisting|minted|{BOXES})\}}" + inside("env", 3000)
        + r"\b(?:I (?:want|need) you to|act as an?\b|return (?:only|your answer)|respond (?:only )?with"
        r"|output (?:only|the following)|do not (?:output|include|explain))" + inside("env", 5000)
        + r"\\end\{(?P=env)\}", re.DOTALL | re.I), 0),
    # Carried over:
    # A prompt written as a JSON field, e.g. "instruction": "You are ...", {"role": "system", "content": ...}
    ("json_prompt_field", re.compile(
        r"[\"“](?:instruction|system_prompt|prompt|system|user|role)[\"”]\s*:\s*[\"“]"), 1200),
    # Undecided since iteration 2:
    # A sentence that introduces a prompt shown next, e.g. "the prompt we used is as follows"
    ("prompt_leadin", re.compile(
        r"(?<![\w(])prompts?(?!\()(?:\s+\w+){0,2}\s+(?:is|are|was|reads)\s+(?:as follows|shown below|given below)",
        re.I), 1200),
]

PROMPT_WORD = re.compile(r"prompt", re.I)
# Whole-line LaTeX comments: commented-out figures and listings are not in the paper.
COMMENT_LINE = re.compile(r"(?m)^[ \t]*%.*$")
LATEX_COMMAND = re.compile(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?")
MIN_BLOCK_TEXT = 80
SAMPLE_SIZE = 20
EXCERPT_CHARS = 6000


def strip_comments(text):
    return COMMENT_LINE.sub("", text)


def detect(text, patterns=PATTERNS):
    """Return (name, start, end) for every match of `patterns` in `text`.

    `text` should already have comments stripped. A whole-block match (trail 0)
    needs at least MIN_BLOCK_TEXT characters of non-command text, so empty boxes
    and listings don't count."""
    hits = []
    for name, rx, trail, *lead in patterns:
        lead = lead[0] if lead else 0
        for m in rx.finditer(text):
            if not trail and len("".join(LATEX_COMMAND.sub("", m.group(0)).split())) < MIN_BLOCK_TEXT:
                continue
            if name in REQUIRE and not REQUIRE[name](m):
                continue
            hits.append((name, max(0, m.start() - lead), min(len(text), m.end() + trail)))
    return hits


def excerpt(text, hits):
    """An excerpt for the review sheet: up to three non-overlapping hits, each
    headed by the pattern that found it, or "prompt" mentions if none."""
    if hits:
        parts, end = [], -1
        for name, s, e in sorted(hits, key=lambda h: h[1]):
            if s < end:
                continue
            parts.append(f"=== [{name}] ===\n" + text[s:min(e, s + EXCERPT_CHARS // 3)])
            end = e
            if len(parts) == 3:
                break
        return "\n\n".join(parts)
    ctx = [text[max(0, m.start() - 200): m.end() + 200] for m in list(PROMPT_WORD.finditer(text))[:3]]
    return "\n---\n".join(ctx) if ctx else "(no mention of 'prompt')"


def write_hits(ds, path):
    """Write every flagged paper's hits to `path`; return the flagged and unflagged row
    indices and the papers each pattern fired on."""
    flagged, unflagged, per_pattern = [], [], {}
    with open(path, "w") as f:
        for i, row in enumerate(ds):
            text = strip_comments(row["latex"] or "")
            hits = detect(text)
            if hits:
                flagged.append(i)
                for name, _, _ in hits:
                    per_pattern[name] = per_pattern.get(name, 0) + 1
                f.write(json.dumps({
                    "id": row["id"], "title": row["title"],
                    "hits": [{"method": n, "start": s, "end": e, "text": text[s:e]} for n, s, e in hits],
                }) + "\n")
            else:
                unflagged.append(i)
    return flagged, unflagged, per_pattern


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--dataset", default="datasets/cs_arxiv_latex_dataset")
    parser.add_argument("--outdir", default="data/prompts/iterations")
    # The frozen detector on papers added after development: hits only, no review sample.
    parser.add_argument("--hits-only", metavar="PATH",
                        help="write the adopted patterns' hits on --dataset to PATH and stop")
    args = parser.parse_args()
    if args.iteration is None and not args.hits_only:
        parser.error("--iteration is required unless --hits-only is given")

    from datasets import load_from_disk

    ds = load_from_disk(args.dataset)
    if args.hits_only:
        os.makedirs(os.path.dirname(args.hits_only), exist_ok=True)
        flagged, unflagged, per_pattern = write_hits(ds, args.hits_only)
        print(json.dumps({"papers": len(ds), "flagged_papers": len(flagged),
                          "hits_per_pattern": per_pattern, "hits": args.hits_only}, indent=1))
        return
    it_dir = os.path.join(args.outdir, f"iter_{args.iteration:02d}")
    os.makedirs(it_dir, exist_ok=True)
    seen_path = os.path.join(args.outdir, "sampled_ids.txt")
    seen = set(open(seen_path).read().split()) if os.path.exists(seen_path) else set()

    flagged, unflagged, per_pattern = write_hits(ds, os.path.join(it_dir, "hits.jsonl"))

    rng = random.Random(args.iteration)
    pos = [i for i in flagged if ds[i]["id"] not in seen]
    neg = [i for i in unflagged if ds[i]["id"] not in seen]
    rng.shuffle(pos)
    rng.shuffle(neg)
    sample = [("flagged", i) for i in pos[:SAMPLE_SIZE]] + [("not_flagged", i) for i in neg[:SAMPLE_SIZE]]

    with open(os.path.join(it_dir, "review.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample", "paper_id", "title", "arxiv_url", "detector_methods",
                    "candidate_patterns_firing", "excerpt", "your_label", "your_notes"])
        for kind, i in sample:
            row = ds[i]
            text = strip_comments(row["latex"] or "")
            hits = detect(text)
            cands = sorted({n for n, _, _ in detect(text, CANDIDATES)})
            w.writerow([kind, row["id"], row["title"], f"https://arxiv.org/pdf/{row['id']}",
                        ";".join(sorted({n for n, _, _ in hits})), ";".join(cands),
                        excerpt(text, hits), "", ""])
    with open(seen_path, "a") as f:
        for _, i in sample:
            f.write(ds[i]["id"] + "\n")

    print(json.dumps({
        "iteration": args.iteration,
        "papers": len(ds),
        "flagged_papers": len(flagged),
        "hits_per_pattern": per_pattern,
        "unflagged_papers": len(unflagged),
        "sampled_flagged": min(SAMPLE_SIZE, len(pos)),
        "sampled_not_flagged": min(SAMPLE_SIZE, len(neg)),
        "review_sheet": os.path.join(it_dir, "review.csv"),
    }, indent=1))


if __name__ == "__main__":
    main()
