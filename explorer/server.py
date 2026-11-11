"""SEPIA explorer: a small local server for the cluster map and the prompt search.

  python server.py [--port 8765] [--pylib DIR] [--no-model]

Serves ./static and the map data, and answers four JSON endpoints:
  GET  /api/status                 model state and dataset size
  GET  /api/prompt?i=N             full record of prompt N (text only when released)
  GET  /api/similar?i=N&k=..       prompts closest to prompt N in embedding space
  GET  /api/find?q=..              prompts whose full text, title, labels or id contain all terms
  GET  /api/suggestions            suggested task types for unassigned prompts (not dataset labels)
  POST /api/search {q, k, mode}    free-text search, "semantic" (Harrier) or "keyword" (TF-IDF)
Filters for the last two: groups=0,3 years=2025,2026 released=1.

Semantic search embeds the query exactly as the dataset prompts were embedded:
LaTeX stripped, template slots masked, the task instruction as prompt, Harrier-OSS-v1-0.6B
at the pinned revision, normalized vectors. Without the model the server still runs, with
keyword search for free text and embedding neighbors for "similar prompts".
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
STATIC = os.path.join(HERE, "static")

EMBEDDING_MODEL = "microsoft/harrier-oss-v1-0.6b"
EMBEDDING_REVISION = "f9b9dc8d367d443f2479d27aa5d8d2850c0774ee"
INSTRUCTION = "Instruct: Identify the software engineering task of prompts written for large language models\nQuery:"
MAX_TOKENS = 1024
MAX_K = 100
# Suggested types: an unassigned prompt gets the type most common among its SUGGEST_K nearest typed
# prompts (its own family excluded) when at least SUGGEST_MIN of them agree.
SUGGEST_K = 10
SUGGEST_MIN = 7

# --- query normalization, the same LaTeX -> prose steps used for the dataset ----------
BRACED = r"\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}"
DROP_WITH_ARGS = re.compile(
    r"\\(?:caption|captionof|label|ref|autoref|cref|Cref|eqref|cite[tpn]?|url|href|"
    r"includegraphics|vspace|hspace|setlength|setcounter|addtocounter|definecolor|color|"
    r"cellcolor|rowcolor|rule|newcommand|renewcommand|providecommand|newtcolorbox|"
    r"newenvironment|tcbset|lstset|lstdefinestyle)\*?(?:\[[^\]]*\])?"
    rf"(?:\s*{BRACED})*")
LAYOUT_ARGS = {"resizebox": 2, "scalebox": 1, "textcolor": 1, "colorbox": 1, "fcolorbox": 2,
               "multirow": 2, "multicolumn": 2, "tabincell": 1, "parbox": 1, "makebox": 0}
DROP_LAYOUT_ARGS = re.compile("|".join(
    rf"\\{name}\*?(?:\[[^\]]*\])*" + rf"(?:\s*{BRACED}){{{n}}}" for name, n in LAYOUT_ARGS.items()))
DROP_ASSIGNMENT = re.compile(r"\\[A-Za-z@]+\s*=\s*-?[\d.]+\s*[a-z]{0,2}")
ENV_TAG = re.compile(rf"\\(?:begin|end)\{{[^}}]*\}}(?:\[[^\]]*\]|{BRACED})*")
COMMAND = re.compile(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?")
ESCAPED = {r"\{": "\x01", r"\}": "\x02", r"\_": "_", r"\#": "#", r"\%": "%",
           r"\&": "&", r"\$": "\x03", r"\textbackslash": "\\"}
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


class Index:
    def __init__(self):
        t = time.time()
        self.points = json.load(open(os.path.join(DATA, "points.json"), encoding="utf-8"))
        self.records = json.load(open(os.path.join(DATA, "records.json"), encoding="utf-8"))
        self.emb = np.load(os.path.join(DATA, "embeddings.npy"), allow_pickle=False).astype(np.float32)
        cols = self.points["cols"]
        self.n = len(self.records)
        self.group = np.array(cols["g"])
        self.released = np.array(cols["r"], dtype=bool)
        years = np.array([p["year"] for p in self.points["papers"]])
        self.year = years[np.array(cols["p"])]
        self.xy = np.stack([np.array(cols["x"]), np.array(cols["y"])], 1)
        fam = [r["family_id"] for r in self.records]
        self.family = np.array([int(f) if f is not None else -1 - i for i, f in enumerate(fam)])
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = [r["text_normalised"] or "" for r in self.records]
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True,
                                     stop_words="english", token_pattern=r"(?u)\b\w[\w\-]+\b")
        self.tfidf_matrix = self.tfidf.fit_transform(texts)
        # Plain-text haystack for the map's highlight box: full released text, title, labels, ids.
        P = self.points
        self.hay = []
        for i, r in enumerate(self.records):
            pa = P["papers"][cols["p"][i]]
            self.hay.append(" ".join([
                r["text_normalised"] or "", pa["title"], P["types"][cols["t"][i]],
                P["clusters"][cols["c"][i]], pa["id"], r["prompt_id"]]).lower())
        self.suggestions = self.suggest()
        self.model = None
        self.model_state = "not loaded"
        self.model_check = None
        self.lock = threading.Lock()
        print(f"index: {self.n} prompts in {time.time() - t:.1f}s", flush=True)

    # --- model -------------------------------------------------------------------
    def load_model(self):
        self.model_state = "loading"
        try:
            from sentence_transformers import SentenceTransformer
            t = time.time()
            model = SentenceTransformer(EMBEDDING_MODEL, revision=EMBEDDING_REVISION, device="cpu",
                                        model_kwargs={"dtype": "float32"})
            model.max_seq_length = MAX_TOKENS
            self.model = model
            # Self-check: re-embed a few released prompts and compare with the stored vectors.
            idx = [i for i in np.flatnonzero(self.released)[:: max(1, self.released.sum() // 6)]][:6]
            vecs = self.encode([self.records[i]["text_normalised"] for i in idx], normalized=True)
            self.model_check = float(np.mean(np.sum(vecs * self.emb[idx], 1)))
            self.model_state = "ready"
            print(f"model ready in {time.time() - t:.0f}s, check cosine {self.model_check:.4f}", flush=True)
        except Exception as e:  # keep serving keyword search
            self.model_state = f"unavailable: {type(e).__name__}: {e}"[:300]
            print("model", self.model_state, flush=True)

    def encode(self, texts, normalized=False):
        texts = texts if normalized else [normalise(t) for t in texts]
        with self.lock:
            return self.model.encode(texts, prompt=INSTRUCTION, batch_size=8,
                                     normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    # --- search ------------------------------------------------------------------
    def mask(self, q):
        m = np.ones(self.n, dtype=bool)
        if q.get("groups"):
            m &= np.isin(self.group, [int(g) for g in str(q["groups"]).split(",") if g != ""])
        if q.get("years"):
            m &= np.isin(self.year, [int(y) for y in str(q["years"]).split(",") if y != ""])
        if str(q.get("released", "0")) in ("1", "true", "True"):
            m &= self.released
        return m

    def top(self, scores, mask, k, exclude=None, min_score=None):
        s = np.where(mask, scores, -np.inf)
        if exclude is not None:
            s[exclude] = -np.inf
        if min_score is not None:
            s[s <= min_score] = -np.inf
        k = min(k, int(np.isfinite(s).sum()))
        if k <= 0:
            return []
        idx = np.argpartition(-s, k - 1)[:k]
        return idx[np.argsort(-s[idx])].tolist()

    def place(self, idx, scores):
        """Where a query falls on the map: the score-weighted mean of its closest results."""
        if not idx:
            return None
        top = idx[:8]
        w = np.exp((scores[top] - scores[top].max()) * 30)
        x, y = (self.xy[top] * w[:, None]).sum(0) / w.sum()
        return [round(float(x), 4), round(float(y), 4)]

    def suggest(self):
        t = np.array(self.points["cols"]["t"])
        unassigned = self.points["types"].index("Unassigned")
        typed, todo = np.flatnonzero(t != unassigned), np.flatnonzero(t == unassigned)
        sims = self.emb[todo] @ self.emb[typed].T
        sims[self.family[todo][:, None] == self.family[typed][None, :]] = -9  # a template's copies do not vote
        nn = np.argpartition(-sims, SUGGEST_K, axis=1)[:, :SUGGEST_K]
        out = {}
        for row, i in enumerate(todo):
            labels, counts = np.unique(t[typed[nn[row]]], return_counts=True)
            j = counts.argmax()
            if counts[j] >= SUGGEST_MIN:
                out[int(i)] = [int(labels[j]), int(counts[j])]
        print(f"suggested types for {len(out)} of {len(todo)} unassigned prompts", flush=True)
        return {"k": SUGGEST_K, "min": SUGGEST_MIN, "items": out}

    def find(self, text):
        """Prompts whose haystack contains every whitespace-separated term of `text`."""
        terms = [t for t in text.lower().split() if t]
        if not terms or sum(map(len, terms)) < 2:
            return {"terms": terms, "results": []}
        return {"terms": terms, "results": [i for i, h in enumerate(self.hay) if all(t in h for t in terms)]}

    def similar(self, i, q):
        k = min(int(q.get("k", 25)), MAX_K)
        scores = self.emb @ self.emb[i]
        idx = self.top(scores, self.mask(q), k, exclude=i)
        return {"mode": "similar", "of": i, "place": [float(v) for v in self.xy[i]],
                "results": [[j, round(float(scores[j]), 4), bool(self.family[j] == self.family[i])] for j in idx]}

    def search(self, body):
        text = (body.get("q") or "").strip()
        if not text:
            return {"error": "empty query"}
        k = min(int(body.get("k", 25)), MAX_K)
        mode = body.get("mode", "semantic")
        mask = self.mask(body)
        t = time.time()
        if mode == "semantic" and self.model is not None:
            v = self.encode([text])[0]
            scores = self.emb @ v
            idx = self.top(scores, mask, k)
            place = self.place(idx, scores)
        else:
            mode = "keyword"
            v = self.tfidf.transform([normalise(text)])
            scores = (self.tfidf_matrix @ v.T).toarray().ravel()
            idx = self.top(scores, mask, k, min_score=0)
            place = self.place(idx, scores) if idx else None
        return {"mode": mode, "took_ms": round((time.time() - t) * 1000), "place": place,
                "results": [[j, round(float(scores[j]), 4), False] for j in idx]}


INDEX = None


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=STATIC, **kw)

    def list_directory(self, path):
        # a public deployment should not list folder contents
        self.send_error(404, "Not found")
        return None

    def end_headers(self):
        # always revalidate the page, script and styles, so an edited file shows up on the next refresh
        if not self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt, *args):
        if args and "/api/" in str(args[0]):
            sys.stderr.write("%s\n" % (fmt % args))

    def send_json(self, obj, status=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/data/points.json":
                return self.send_file(os.path.join(DATA, "points.json"), "application/json")
            if url.path == "/api/status":
                return self.send_json({"model": INDEX.model_state, "check": INDEX.model_check, "n": INDEX.n})
            if url.path == "/api/prompt":
                i = int(q["i"])
                return self.send_json(INDEX.records[i])
            if url.path == "/api/similar":
                return self.send_json(INDEX.similar(int(q["i"]), q))
            if url.path == "/api/suggestions":
                return self.send_json(INDEX.suggestions)
            if url.path == "/api/find":
                return self.send_json(INDEX.find(q.get("q", "")[:200]))
        except (KeyError, ValueError, IndexError) as e:
            return self.send_json({"error": f"bad request: {e}"}, 400)
        if url.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        return super().do_GET()

    def do_POST(self):
        url = urlparse(self.path)
        if url.path != "/api/search":
            return self.send_json({"error": "not found"}, 404)
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(min(n, 200_000)) or b"{}")
            return self.send_json(INDEX.search(body))
        except (ValueError, KeyError) as e:
            return self.send_json({"error": f"bad request: {e}"}, 400)

    def send_file(self, path, ctype):
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    global INDEX
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8765)))
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    ap.add_argument("--pylib", help="extra folder with Python packages (e.g. sentence-transformers)")
    ap.add_argument("--no-model", action="store_true", help="keyword search only")
    args = ap.parse_args()
    if args.pylib:
        sys.path.insert(0, args.pylib)
    INDEX = Index()
    if args.no_model:
        INDEX.model_state = "disabled"
    else:
        threading.Thread(target=INDEX.load_model, daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"SEPIA explorer on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
