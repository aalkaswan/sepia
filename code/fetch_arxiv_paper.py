#!/usr/bin/env python3
"""Find an arXiv paper by identifier, and fetch it from arXiv if the local copy can't supply it.

Lookup order:
  1. The local arxiv-complete dataset (metadata and versions configs), read inside a
     container with pyarrow, so nothing is installed on the host.
  2. If the paper is absent, or newer than the dataset snapshot, download the PDF from
     arxiv.org/pdf/, following the gap-filling method the dataset's README describes:
     the /pdf/ path, the robots.txt Crawl-delay, one retry per failure, and a recorded
     SHA-256 digest. Requests are spaced by the Crawl-delay; the creators' faster pace
     is deliberately not reproduced.

Examples:
  fetch_arxiv_paper.py 1706.03762
  fetch_arxiv_paper.py 2609.01234v1                # too new for the snapshot -> fetched
  fetch_arxiv_paper.py gr-qc/9408005 --no-fetch    # local lookup only
  fetch_arxiv_paper.py 1706.03762 --fetch-only     # skip local lookup
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

REPO = os.path.dirname(os.path.abspath(__file__))
# The arXiv Complete Corpus snapshot: https://huggingface.co/datasets/secemp9/arxiv-complete
# Its metadata and versions configs (Parquet) are expected here; paper_text holds the TeX.
DATA_DIR = os.path.join(REPO, "data", "arxiv-complete")
FETCH_DIR = os.path.join(REPO, "data", "fetched")

# Latest submission date in the arxiv-complete snapshot (README, "Snapshot and coverage").
SNAPSHOT_LATEST = datetime.date(2026, 8, 27)

CRAWL_DELAY = 15  # seconds, from https://arxiv.org/robots.txt
API_DELAY = 3     # seconds between export.arxiv.org API calls
USER_AGENT = "arxiv-paper-fetch/1.0 (single-paper lookup; polite)"
CONTAINER_IMAGE = "python:3.12-slim"

NEW_ID = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
OLD_ID = re.compile(r"^[a-z\-]+(\.[A-Za-z]{2})?/\d{7}(v\d+)?$")

_last_request = 0.0


def normalize_id(raw):
    """Accept 'arXiv:1706.03762', a URL, or a bare identifier with an optional version."""
    s = raw.strip()
    s = re.sub(r"^arxiv:", "", s, flags=re.IGNORECASE)
    m = re.search(r"arxiv\.org/(?:abs|pdf)/(.+?)(?:\.pdf)?$", s)
    if m:
        s = m.group(1)
    if NEW_ID.match(s) or OLD_ID.match(s):
        return s
    sys.exit(f"Not a recognised arXiv identifier: {raw!r}")


def base_id(pid):
    return re.sub(r"v\d+$", "", pid)


def throttle(delay):
    """Block until `delay` seconds have passed since the last request to arXiv."""
    global _last_request
    wait = delay - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()


def local_lookup(pid):
    """Query the local dataset in a container. Returns a dict, or None if the dataset is absent."""
    if not os.path.isdir(os.path.join(DATA_DIR, "metadata")):
        print("local: dataset metadata not found, skipping local lookup", file=sys.stderr)
        return None
    script = (
        "import os, json\n"
        "import pyarrow.dataset as ds\n"
        "pid = os.environ['ARXIV_ID']\n"
        "out = {}\n"
        "for cfg, cols in {\n"
        "    'metadata': ['paper_id', 'title', 'authors', 'categories',\n"
        "                 'first_version_date', 'latest_version_date', 'arxiv_abs_url'],\n"
        "    'versions': ['paper_id', 'version', 'version_date', 'has_pdf', 'has_ps',\n"
        "                 'has_html', 'reported_size_kb', 'is_metadata_only'],\n"
        "}.items():\n"
        "    t = ds.dataset('/data/' + cfg, format='parquet').to_table(\n"
        "        columns=cols, filter=ds.field('paper_id') == pid)\n"
        "    out[cfg] = t.to_pylist()\n"
        "print(json.dumps(out, default=str))\n"
    )
    # The script travels in an environment variable so the shell never re-parses it.
    cmd = [
        "docker", "run", "--rm",
        "-e", f"ARXIV_ID={base_id(pid)}",
        "-e", f"PY={script}",
        "-v", f"{DATA_DIR}:/data:ro",
        CONTAINER_IMAGE,
        "sh", "-c", "pip install -q pyarrow==18.1.0 && python -I -c \"$PY\"",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except FileNotFoundError:
        print("local: docker not available, skipping local lookup", file=sys.stderr)
        return None
    if result.returncode != 0:
        print("local: container lookup failed:\n" + result.stderr.strip(), file=sys.stderr)
        return None
    return json.loads(result.stdout.strip().splitlines()[-1])


def remote_metadata(pid):
    """Fetch title and dates from the arXiv API. Returns None if arXiv does not know the ID."""
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({"id_list": base_id(pid)})
    throttle(API_DELAY)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        root = ET.fromstring(resp.read())
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entry = root.find("a:entry", ns)
    if entry is None or entry.find("a:title", ns) is None:
        return None
    title = " ".join(entry.find("a:title", ns).text.split())
    if title == "Error":
        return None
    return {
        "title": title,
        "authors": [a.find("a:name", ns).text for a in entry.findall("a:author", ns)],
        "published": entry.find("a:published", ns).text,
        "updated": entry.find("a:updated", ns).text,
        "primary_category": entry.find("{http://arxiv.org/schemas/atom}primary_category").get("term"),
    }


def fetch_pdf(pid, out_dir):
    """Download the PDF using the README's gap-filling method. Returns the saved path or None."""
    url = f"https://arxiv.org/pdf/{pid}"
    for attempt in (1, 2):
        throttle(CRAWL_DELAY)
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                ctype = resp.headers.get("Content-Type", "")
                body = resp.read()
        except urllib.error.HTTPError as e:
            print(f"fetch: attempt {attempt} got HTTP {e.code} for {url}", file=sys.stderr)
            continue
        except urllib.error.URLError as e:
            print(f"fetch: attempt {attempt} failed: {e.reason}", file=sys.stderr)
            continue
        # The README's categories of non-success: 404 HTML pages, empty 200s and 500s.
        if "pdf" not in ctype or not body.startswith(b"%PDF") or b"%%EOF" not in body[-1024:]:
            print(f"fetch: attempt {attempt} returned no usable PDF (type {ctype}, {len(body)} bytes)",
                  file=sys.stderr)
            continue
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{pid.replace('/', '_')}.pdf")
        with open(path, "wb") as f:
            f.write(body)
        return path, hashlib.sha256(body).hexdigest(), len(body), url
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arxiv_id", help="arXiv identifier, e.g. 1706.03762, 2609.01234v1, gr-qc/9408005")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--no-fetch", action="store_true", help="only look up locally; never download")
    group.add_argument("--fetch-only", action="store_true", help="skip the local lookup")
    args = ap.parse_args()

    pid = normalize_id(args.arxiv_id)
    report = {"id": pid, "local": None, "remote": None, "fetched": None}

    if not args.fetch_only:
        report["local"] = local_lookup(pid)
        found = bool(report["local"] and report["local"]["metadata"])
        if found:
            meta = report["local"]["metadata"][0]
            latest = str(meta["latest_version_date"] or "")[:10]
            print(f"local: found {base_id(pid)}: {meta['title']}", file=sys.stderr)
            if latest > SNAPSHOT_LATEST.isoformat():
                print("local: has a version newer than the snapshot, so fetching the latest from arXiv",
                      file=sys.stderr)
                args.fetch_only = True
        elif report["local"] is not None:
            print(f"local: {base_id(pid)} not in the snapshot", file=sys.stderr)
            if args.no_fetch:
                print(json.dumps(report, indent=2, default=str))
                return 1

    if args.no_fetch:
        print(json.dumps(report, indent=2, default=str))
        return 0 if report["local"] and report["local"]["metadata"] else 1

    report["remote"] = remote_metadata(pid)
    if report["remote"] is None:
        print(f"remote: arXiv has no record of {pid}", file=sys.stderr)
        print(json.dumps(report, indent=2, default=str))
        return 1
    print(f"remote: {report['remote']['title']} ({report['remote']['published'][:10]})", file=sys.stderr)

    result = fetch_pdf(pid, os.path.join(FETCH_DIR, base_id(pid)))
    if result is None:
        print(f"fetch: could not download a PDF for {pid}", file=sys.stderr)
        print(json.dumps(report, indent=2, default=str))
        return 1
    path, sha, size, url = result
    record = {"source_url": url, "sha256": sha, "size": size,
              "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    report["fetched"] = dict(record, path=path)
    with open(os.path.splitext(path)[0] + ".json", "w") as f:
        json.dump({"id": pid, "remote": report["remote"], **record}, f, indent=2)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
