"""Reconcile the two annotators' prompt lists into one gold list of prompts.

The annotation page's Reconcile view shows, for every gold paper with prompts,
both annotators' entries. Entries in the same group are one prompt; each group
is kept as a prompt or dropped as not a prompt.

  seed    write the page's database documents to data/gold/reconcile_seed/:
          reconcile/<paper_id> (both lists, read-only on the page) and
          decisions/<paper_id> (groups, pre-filled with the automatic matches)
  export  read the decisions exported from the page to
          data/gold/reconcile/decisions/<paper_id>.json and write
          data/gold/prompts.csv, one row per gold prompt

Usage: python3 gold_reconcile.py seed|export
"""
import csv
import glob
import json
import os
import sys

import gold_agreement as g

SEED_DIR = os.path.join(g.GOLD_DIR, "reconcile_seed")
DECISIONS = os.path.join(g.GOLD_DIR, "reconcile", "decisions")


def gold_papers():
    """Paper ids whose resolved label is "has prompts", in annotation order, and the annotations."""
    sample, ann = g.load_annotators()
    adj = {}
    if os.path.exists(os.path.join(g.GOLD_DIR, "adjudicated.csv")):
        adj = {r["paper_id"]: r["has_prompts"].strip() == "1"
               for r in csv.DictReader(open(os.path.join(g.GOLD_DIR, "adjudicated.csv"), encoding="utf-8"))
               if r["has_prompts"].strip()}
    first = next(iter(ann.values()))
    rows = list(csv.DictReader(open(os.path.join(g.GOLD_DIR, "sample.csv"), encoding="utf-8")))
    return [r for r in rows if adj.get(r["paper_id"], first[r["paper_id"]]["has_prompts"])], ann


def same_prompt(x, y):
    return g.matches(x, [g.norm(y)]) or g.matches(y, [g.norm(x)])


def seed():
    rows, ann = gold_papers()
    (ua, la), (ub, lb) = ann.items()
    os.makedirs(SEED_DIR, exist_ok=True)
    writes = []
    for order, r in enumerate(rows):
        pid = r["paper_id"]
        items = []
        for tag, labels in (("A", la), ("B", lb)):
            for i, q in enumerate(labels[pid]["prompts"]):
                items.append({"id": f"{tag}{i}", "by": tag, "where": q.get("where", ""), "form": q.get("form", ""),
                              "locator": q.get("locator", ""), "first_words": q.get("first_words", "")})
        # Each entry starts in its own group; automatic matches share the A entry's group.
        group = {it["id"]: it["id"] for it in items}
        auto = set()
        taken = set()
        for a in (it for it in items if it["by"] == "A"):
            for b in (it for it in items if it["by"] == "B"):
                if b["id"] not in taken and same_prompt(a["first_words"], b["first_words"]):
                    group[b["id"]] = a["id"]
                    taken.add(b["id"])
                    auto.update((a["id"], b["id"]))
                    break
        notes = {"A": la[pid].get("notes", ""), "B": lb[pid].get("notes", "")}
        rec = {"paper_id": pid, "title": r["title"], "pdf_url": r["pdf_url"], "order": order,
               "annotators": {"A": ua, "B": ub}, "items": items, "notes": notes}
        dec = {"paper_id": pid, "done": False, "note": "",
               "items": {it["id"]: {"g": group[it["id"]], "verdict": "prompt" if it["id"] in auto else None,
                                    "auto": it["id"] in auto} for it in items}}
        for coll, doc in (("reconcile", rec), ("decisions", dec)):
            path = os.path.join(SEED_DIR, f"{coll}_{pid}.json")
            json.dump(doc, open(path, "w"), ensure_ascii=False)
            writes.append({"op": "set", "collection": coll, "doc_id": pid, "file_path": os.path.abspath(path)})
    for i in range(0, len(writes), 50):
        json.dump(writes[i:i + 50], open(os.path.join(SEED_DIR, f"writes_{i // 50}.json"), "w"), indent=1)
    print(f"{len(rows)} papers, {len(writes)} documents, {sum(len(json.load(open(w['file_path']))['items']) for w in writes if w['collection'] == 'reconcile')} entries")


def export():
    rows, ann = gold_papers()
    (ua, la), (ub, lb) = ann.items()
    out, unfinished = [], []
    for r in rows:
        pid = r["paper_id"]
        path = os.path.join(DECISIONS, f"{pid}.json")
        dec = json.load(open(path)) if os.path.exists(path) else None
        if not dec or not dec.get("done"):
            unfinished.append(pid)
            continue
        entries = {f"A{i}": q for i, q in enumerate(la[pid]["prompts"])}
        entries.update({f"B{i}": q for i, q in enumerate(lb[pid]["prompts"])})
        groups = {}
        for iid, d in dec["items"].items():
            groups.setdefault(d["g"], []).append((iid, d))
        for n, (gid, members) in enumerate(sorted(groups.items())):
            if any(d["verdict"] != "prompt" for _, d in members):
                continue
            ids = sorted(i for i, _ in members)
            # The longest first-words entry stands for the prompt when matching it to the detector.
            rep = max((entries[i] for i in ids), key=lambda q: len(q.get("first_words", "")))
            out.append({"paper_id": pid, "prompt": n, "by_a": int(any(i.startswith("A") for i in ids)),
                        "by_b": int(any(i.startswith("B") for i in ids)), "entries": ";".join(ids),
                        "where": rep.get("where", ""), "form": rep.get("form", ""),
                        "locator": rep.get("locator", ""), "first_words": rep.get("first_words", "")})
    with open(os.path.join(g.GOLD_DIR, "prompts.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]) if out else ["paper_id"])
        w.writeheader()
        w.writerows(out)
    print(f"{len(out)} gold prompts from {len(rows) - len(unfinished)} reconciled papers; not reconciled yet: {unfinished}")


if __name__ == "__main__":
    {"seed": seed, "export": export}[sys.argv[1]]()
