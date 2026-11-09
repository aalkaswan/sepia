"""Re-extract the prompt texts that the release gives as pointers only.

Papers not under a Creative Commons licence may not be redistributed, so their prompts in
prompts.jsonl carry `text: null` with a pointer: arXiv identifier, version, position, the
offsets in the assembled source and a SHA-256 of the text. This script downloads each such
paper's e-print from arXiv (or reads it from --sources), assembles the TeX the way the
pipeline did, runs the same detector and fills in every prompt whose digest matches.

  python reextract.py --download --email you@example.org        # all pointer-only prompts
  python reextract.py --download --ids 2401.01234 2402.04567
  python reextract.py --sources my_eprints/                      # offline, no downloads

Writes prompts_full.jsonl next to prompts.jsonl, with `text`, `text_normalised` and
`reextracted` per prompt: "matched" (same version and position), "found" (elsewhere in the
same version's files), "found in vN" (an earlier version), "not found" or "no source". Downloads follow arXiv's
robots.txt Crawl-delay and are cached in --sources; nothing is ever uploaded. Python 3.10+
standard library only.
"""
import argparse
import hashlib
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_added_dataset import assemble, source_files
from collect_prompts import detect, strip_comments
from task_clusters import merge_hits, normalise

SRC_URL = "https://export.arxiv.org/src/{id}v{version}"
CRAWL_DELAY = 15  # seconds, from https://arxiv.org/robots.txt
_last = 0.0


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def eprint(arxiv_id, version, sources, download, email):
    """The e-print archive's bytes, from the cache or arXiv; None if unavailable."""
    path = os.path.join(sources, f"{arxiv_id.replace('/', '_')}v{version}")
    if os.path.exists(path):
        return open(path, "rb").read()
    if not download:
        return None
    global _last
    agent = "promptminer-reextract/1.0" + (f" (mailto:{email})" if email else "")
    req = urllib.request.Request(SRC_URL.format(id=arxiv_id, version=version), headers={"User-Agent": agent})
    # One retry per failure, each request spaced by the Crawl-delay.
    for attempt in (1, 2):
        time.sleep(max(0.0, _last + CRAWL_DELAY - time.time()))
        try:
            data = urllib.request.urlopen(req, timeout=120).read()
            break
        except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
            print(f"{arxiv_id}v{version}: {e!r} (attempt {attempt})", file=sys.stderr)
            data = None
        finally:
            _last = time.time()
    if data is None:
        return None
    os.makedirs(sources, exist_ok=True)
    open(path, "wb").write(data)
    return data


def blocks_of(data):
    """The detector's blocks in one TeX text, as the pipeline built them."""
    text = strip_comments(data)
    return merge_hits([{"method": n, "start": s, "end": e, "text": text[s:e]} for n, s, e in detect(text)])


def candidates(data):
    """(blocks of the assembled TeX, {digest: text} of every block found).

    Besides the assembled TeX, each TeX file is scanned on its own: the snapshot was
    assembled by its own builder, which resolved some \\input commands differently."""
    files = source_files(data)
    latex, _ = assemble(files)
    assembled = blocks_of(latex) if latex else []
    found = {sha256(b["raw"]): b["raw"] for b in assembled}
    for path, text in sorted(files.items()):
        if path.lower().endswith((".tex", ".ltx")):
            for b in blocks_of(text):
                found.setdefault(sha256(b["raw"]), b["raw"])
    return assembled, found


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", default=os.path.join(here, "..", "prompts.jsonl"))
    parser.add_argument("--out", help="default: prompts_full.jsonl next to --prompts")
    parser.add_argument("--sources", default="eprints", help="cache of downloaded e-prints")
    parser.add_argument("--download", action="store_true", help="fetch e-prints missing from --sources from arXiv")
    parser.add_argument("--email", help="contact address sent to arXiv in the User-Agent")
    parser.add_argument("--ids", nargs="*", help="only these arXiv identifiers")
    parser.add_argument("--older-versions", action=argparse.BooleanOptionalAction, default=True,
                        help="look in earlier versions for prompts the recorded version lacks (snapshot "
                             "sources are unversioned)")
    args = parser.parse_args()

    records = [json.loads(line) for line in open(args.prompts, encoding="utf-8")]
    todo = {}
    for r in records:
        if r["text"] is None and (not args.ids or r["arxiv_id"] in args.ids):
            todo.setdefault((r["arxiv_id"], r["version"]), []).append(r)
    status = {}
    for n, ((pid, version), rs) in enumerate(sorted(todo.items()), 1):
        pending, any_source = list(rs), False
        for v in range(version, 0 if args.older_versions else version - 1, -1):
            data = eprint(pid, v, args.sources, args.download, args.email)
            if data is None:
                continue
            any_source = True
            assembled, found = candidates(data)
            for r in list(pending):
                pos, digest = r["position"], r["text_sha256"]
                if v == version and pos < len(assembled) and sha256(assembled[pos]["raw"]) == digest:
                    r["reextracted"] = "matched"
                elif digest in found:
                    r["reextracted"] = "found" if v == version else f"found in v{v}"
                else:
                    continue
                r["text"] = found[digest]
                r["text_normalised"] = normalise(r["text"])
                pending.remove(r)
            if not pending:
                break
        for r in pending:
            r["reextracted"] = "not found" if any_source else "no source"
        for r in rs:
            status[r["reextracted"]] = status.get(r["reextracted"], 0) + 1
        print(f"[{n}/{len(todo)}] {pid}v{version}: " + ", ".join(r["reextracted"] for r in rs), file=sys.stderr)

    out = args.out or os.path.join(os.path.dirname(args.prompts), "prompts_full.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"papers": len(todo), **status, "written": out}, indent=1))


if __name__ == "__main__":
    main()
