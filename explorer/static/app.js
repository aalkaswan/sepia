/* SEPIA explorer: cluster map + prompt search. Data from /data/points.json, search via /api/*. */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (n) => n.toLocaleString("en-US");
const UNASSIGNED = 8;
const KBD = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent) ? "⌘↵" : "Ctrl ↵";
const RESULT_SIZES = [10, 30, 50, 100];
const EXAMPLES = [
  ["Code Review", "You are a senior developer. Review the following code diff and list potential bugs."],
  ["Unit Tests", "Generate JUnit tests for the following Java method."],
  ["LLM As Judge", "Act as a judge and rate the generated code summary from 1 to 5."],
  ["Bug Localization", "Given a bug report and the repository files, identify the files that need to be changed."],
  ["Requirements", "Classify the following requirement as functional or non-functional."],
  ["Code Translation", "Translate this C function into idiomatic Rust."],
];

const S = {
  colorBy: "group", focus: null, hidden: new Set(), hiddenTypes: new Set(),
  years: new Set(), releasedOnly: false, find: "", findSet: null, labels: true, selected: null, suggest: false,
  model: "loading", mode: "semantic", lastSearch: null, k: 30,
};
let D = null;       // points.json
let N = 0;
let P = {};         // per-point arrays
let C = {};         // theme colors

/* ---------- data ---------- */
async function load() {
  [D, D.sugg] = await Promise.all([
    fetch("/data/points.json").then((r) => r.json()),
    fetch("/api/suggestions").then((r) => r.json()).catch(() => ({ items: {}, k: 10, min: 7 })),
  ]);
  D.suggest = new Map(Object.entries(D.sugg.items).map(([i, v]) => [+i, v]));
  const c = D.cols;
  N = c.x.length;
  P = { x: c.x, y: c.y, g: c.g, t: c.t, c: c.c, p: c.p, r: c.r, f: c.f, w: c.w };
  P.year = c.p.map((pi) => D.papers[pi].year);
  D.years = [...new Set(P.year)].sort();
  S.years = new Set(D.years);
  // counts
  D.groupCount = D.groups.map(() => 0);
  D.typeCount = D.types.map(() => 0);
  for (let i = 0; i < N; i++) { D.groupCount[P.g[i]]++; D.typeCount[P.t[i]]++; }
  D.groupTypes = D.groups.map((_, g) =>
    D.types.map((_, t) => t).filter((t) => D.typeGroup[t] === g).sort((a, b) => D.typeCount[b] - D.typeCount[a]));
  $$(".n-prompts").forEach((e) => (e.textContent = fmt(N)));
  // the numbers on the About page
  const stat = { prompts: fmt(N), papers: fmt(D.papers.length), types: String(D.types.filter((t) => t !== "Unassigned").length),
    years: `${D.years[0]}–${D.years[D.years.length - 1]}` };
  $$("[data-stat]").forEach((e) => (e.textContent = stat[e.dataset.stat]));
}

function readTheme() {
  const cs = getComputedStyle(document.documentElement);
  const v = (n) => cs.getPropertyValue(n).trim();
  C = {
    surface: v("--surface"), page: v("--page"), ink: v("--ink"), ink2: v("--ink-2"), muted: v("--muted"), line: v("--line"),
    g: D.groups.map((_, i) => v(`--g${i}`)), t: [0, 1, 2, 3].map((i) => v(`--t${i}`)), y: [0, 1, 2, 3].map((i) => v(`--y${i}`)),
    dark: document.documentElement.dataset.theme ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches,
  };
}
function rgba(hex, a) {
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 3 ? h.split("").map((x) => x + x).join("") : h, 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}
const groupColor = (g) => C.g[g];
function typeSlot(t) { const list = D.groupTypes[D.typeGroup[t]]; return Math.min(list.indexOf(t), 3); }
function yearColor(y) { const k = D.years.indexOf(y); return C.y[Math.max(0, k - (D.years.length - 4))] || C.y[0]; }
function licenseShort(url) {
  if (!url) return "unknown license";
  if (url.includes("creativecommons.org/publicdomain")) return "CC0 1.0";
  const m = url.match(/licenses\/([a-z-]+)\/([\d.]+)/);
  if (url.includes("creativecommons") && m) return `CC ${m[1].toUpperCase()} ${m[2]}`;
  if (url.includes("arxiv.org/licenses")) return "arXiv license";
  return "other license";
}
const arxivUrl = (pa) => `https://arxiv.org/abs/${pa.id}v${pa.v}`;
function authorsShort(pa) {
  if (!pa.authors?.length) return "";
  const a = pa.authors.slice(0, 2).join(", ");
  return pa.nauthors > 2 ? `${a} et al.` : pa.authors.join(" and ");
}

/* ---------- visibility ---------- */
function visible(i) {
  if (S.hidden.has(P.g[i])) return false;
  if (S.hiddenTypes.has(P.t[i])) return false;
  if (!S.years.has(P.year[i])) return false;
  if (S.releasedOnly && !P.r[i]) return false;
  return true;
}

/* ---------- map (deck.gl: smooth zoom, inertia, animated transitions) ---------- */
const MAP = { el: null, deck: null, idx: [], ext: null, labels: [], view: null, base: [], fit: null };
const MONO = "JetBrains Mono, ui-monospace, Menlo, monospace";
function rgb(hex) {
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 3 ? h.split("").map((x) => x + x).join("") : h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function pointStyle(i) {
  // [rgba, radius in px, draw layer]
  if (S.colorBy === "year") return [[...rgb(yearColor(P.year[i])), 225], 3.4, 1];
  if (S.focus !== null) {
    if (P.g[i] !== S.focus) return [[...rgb(C.muted), C.dark ? 64 : 52], 2.5, 0];
    return [[...rgb(C.t[typeSlot(P.t[i])]), 235], 3.9, 2];
  }
  if (P.g[i] === UNASSIGNED) return [[...rgb(C.g[UNASSIGNED]), C.dark ? 150 : 170], 2.7, 0];
  return [[...rgb(groupColor(P.g[i])), 225], 3.4, 1];
}
function computeOrder() {
  const layers = [[], [], []];
  for (let i = 0; i < N; i++) if (visible(i)) layers[pointStyle(i)[2]].push(i);
  let order = layers.flat();
  // by year, draw the newest (and largest) year first so the few older prompts stay on top
  if (S.colorBy === "year") order = order.sort((a, b) => P.year[b] - P.year[a] || a - b);
  MAP.idx = order;
}
function extent() {
  if (!MAP.ext) {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let i = 0; i < N; i++) { x0 = Math.min(x0, P.x[i]); x1 = Math.max(x1, P.x[i]); y0 = Math.min(y0, P.y[i]); y1 = Math.max(y1, P.y[i]); }
    MAP.ext = { x0, x1, y0, y1 };
  }
  return MAP.ext;
}
function fitView() {
  const { x0, x1, y0, y1 } = extent();
  const w = MAP.el.clientWidth || 900, h = MAP.el.clientHeight || 600;
  const noteEl = $("#map-note");  // keep the note card, when shown, clear of the points
  const note = noteEl && !noteEl.hidden && noteEl.offsetHeight ? noteEl.offsetHeight + 24 : 16;
  const zoom = Math.log2(Math.min(w / ((x1 - x0) * 1.08), (h - note) / ((y1 - y0) * 1.06)));
  const cy = (y0 + y1) / 2 - note / 2 / 2 ** zoom;
  MAP.fit = { target: [(x0 + x1) / 2, cy, 0], zoom, minZoom: zoom - 1.5, maxZoom: zoom + 6 };
  return MAP.fit;
}
const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
function flyTo(view, ms = 750) {
  if (!ms) MAP.view = { ...view };  // an instant jump fires no view-state callback
  MAP.deck.setProps({
    initialViewState: { ...view, transitionDuration: ms, transitionEasing: easeInOut,
      transitionInterpolator: new deck.LinearInterpolator(["target", "zoom"]) },
  });
}

function findCount(shown) {
  const el = $("#find-count");
  if (!S.findSet) { el.hidden = true; return; }
  el.hidden = false;
  const total = S.findSet.size;
  if (!total) { el.innerHTML = `<span>No prompt contains “${esc(S.find)}”.</span>`; return; }
  const off = total - shown;
  el.innerHTML = `<span class="ring"></span><span><strong>${fmt(total)}</strong> ${total === 1 ? "prompt" : "prompts"} highlighted${off ? ` <span class="muted">(${fmt(off)} hidden by filters)</span>` : ""}</span>
    <button class="link" id="find-clear">Clear</button>`;
  $("#find-clear").onclick = () => { $("#map-find").value = ""; setFind(""); };
}
let findSeq = 0;
async function setFind(value) {
  S.find = value.trim();
  const seq = ++findSeq;
  if (S.find.replace(/\s/g, "").length < 2) { S.findSet = null; drawMap(); return; }
  const res = await fetch(`/api/find?q=${encodeURIComponent(S.find)}`).then((r) => r.json());
  if (seq !== findSeq) return;  // a newer query is on its way
  S.findSet = new Set(res.results);
  drawMap();
}

/* labels: one pill per task type at its densest blob, edged in the type's color */
const LABEL_NAMES = {
  "Debugging, fault localization and repair": "Debugging & repair",
  "Code review, refactoring and optimization": "Code review & refactoring",
  "Data synthesis, extraction and annotation": "Data synthesis & annotation",
  "Environment setup, builds and shell": "Builds & environment setup",
  "Formal specification and verification": "Formal verification",
  "Software design and modelling": "Design & modelling",
  "Code understanding and analysis": "Code understanding",
  "Code translation and migration": "Code translation",
  "Security and binary analysis": "Security & binary analysis",
  "LLM safety and privacy": "LLM safety & privacy",
};
const shortType = (s) => LABEL_NAMES[s] || s;
function computeLabels() {
  MAP.labels = [];
  if (!S.labels || S.colorBy === "year") return;
  const types = S.focus !== null ? D.groupTypes[S.focus] : D.types.map((_, t) => t).filter((t) => D.typeGroup[t] !== UNASSIGNED);
  const members = new Map(types.map((t) => [t, [[], []]]));
  for (const i of MAP.idx) { const m = members.get(P.t[i]); if (m) { m[0].push(P.x[i]); m[1].push(P.y[i]); } }
  for (const t of types) {
    if (S.hiddenTypes.has(t) || S.hidden.has(D.typeGroup[t])) continue;
    const [xs, ys] = members.get(t);
    if (xs.length < 12) continue;
    const color = S.focus !== null ? C.t[typeSlot(t)] : groupColor(D.typeGroup[t]);
    MAP.labels.push({ pos: densest(xs, ys), text: S.focus !== null ? D.types[t] : shortType(D.types[t]), color, n: xs.length });
  }
  MAP.labels.sort((a, b) => b.n - a.n);  // larger types win when labels collide
}
function placedLabels() {
  const v = MAP.view || MAP.fit;
  if (!v || !MAP.labels.length) return [];
  const scale = 2 ** v.zoom, w = MAP.el.clientWidth, h = MAP.el.clientHeight;
  const size = S.focus !== null ? 13 : 12;
  const boxes = [], out = [];
  for (const l of MAP.labels) {
    const sx = (l.pos[0] - v.target[0]) * scale + w / 2, sy = h / 2 - (l.pos[1] - v.target[1]) * scale;
    const bw = l.text.length * size * 0.61 + 18, bh = size + 12;
    const b = [sx - bw / 2, sy - bh / 2, sx + bw / 2, sy + bh / 2];
    if (boxes.some((o) => !(b[2] < o[0] || b[0] > o[2] || b[3] < o[1] || b[1] > o[3]))) continue;
    boxes.push(b); out.push(l);
  }
  return out;
}
function densest(xs, ys) {
  const G = 36, { x0, x1, y0, y1 } = extent();
  const cell = (v, a, b) => Math.min(G - 1, Math.max(0, Math.floor(((v - a) / (b - a)) * G)));
  const grid = new Map();
  xs.forEach((x, k) => { const key = cell(x, x0, x1) * G + cell(ys[k], y0, y1); grid.set(key, (grid.get(key) || 0) + 1); });
  let best = -1, bestKey = 0;
  for (const [key] of grid) {
    const gx = Math.floor(key / G), gy = key % G;
    let s = 0;
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) s += grid.get((gx + dx) * G + gy + dy) || 0;
    if (s > best) { best = s; bestKey = key; }
  }
  const bx = Math.floor(bestKey / G), by = bestKey % G;
  const px = [], py = [];
  xs.forEach((x, k) => { if (Math.abs(cell(x, x0, x1) - bx) <= 1 && Math.abs(cell(ys[k], y0, y1) - by) <= 1) { px.push(x); py.push(ys[k]); } });
  return [median(px), median(py)];
}
function median(a) { const s = [...a].sort((x, y) => x - y); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; }

/* suggested types for unassigned prompts: a nearest-neighbor vote, shown as hollow rings, never as labels */
function suggestion(i) { return P.g[i] === UNASSIGNED ? D.suggest.get(i) || null : null; }
function suggestText(i, lead) {
  const s = suggestion(i);
  return s ? `${lead}suggested: ${esc(D.types[s[0]])} (${s[1]}/${D.sugg.k})` : "";
}
function suggestBox(i) {
  const s = suggestion(i);
  if (!s) return "";
  const g = D.typeGroup[s[0]];
  return `<div class="suggest" style="--c:${groupColor(g)}"><div class="s-head"><span class="sw"></span>Suggested: ${esc(D.types[s[0]])}</div>
    <p class="hint">${s[1]} of its ${D.sugg.k} nearest typed prompts are ${esc(D.types[s[0]])} (${esc(D.groups[g])}). This is a suggestion from the explorer, not a label in the dataset.</p></div>`;
}
function suggestedPoints() {
  if (!S.suggest || S.colorBy === "year" || S.hidden.has(UNASSIGNED)) return [];
  return MAP.idx.filter((i) => {
    const s = suggestion(i);
    if (!s || S.hidden.has(D.typeGroup[s[0]]) || S.hiddenTypes.has(s[0])) return false;
    return S.focus === null || D.typeGroup[s[0]] === S.focus;
  });
}
function suggestColor(i) {
  const t = suggestion(i)[0];
  return [...rgb(S.focus !== null ? C.t[typeSlot(t)] : groupColor(D.typeGroup[t])), 235];
}
function baseLayers() {
  const key = [S.colorBy, S.focus, C.dark, C.surface, MAP.idx.length, [...S.hiddenTypes].join("."), S.suggest].join("|");
  const pos = (i) => [P.x[i], P.y[i]];
  const hi = S.findSet ? MAP.idx.filter((i) => S.findSet.has(i)) : [];
  findCount(hi.length);
  const ink = [...rgb(C.ink), 255];
  return [
    new deck.ScatterplotLayer({
      id: "points", data: MAP.idx, getPosition: pos, getFillColor: (i) => pointStyle(i)[0], getRadius: (i) => pointStyle(i)[1],
      radiusUnits: "pixels", pickable: true, autoHighlight: true, highlightColor: [...rgb(C.ink), 150],
      updateTriggers: { getFillColor: key, getRadius: key },
    }),
    new deck.ScatterplotLayer({
      id: "suggested", data: suggestedPoints(), getPosition: pos, getRadius: 3.9, radiusUnits: "pixels",
      filled: false, stroked: true, getLineColor: (i) => suggestColor(i), lineWidthUnits: "pixels", getLineWidth: 1.6,
      pickable: true, autoHighlight: true, highlightColor: [...rgb(C.ink), 150],
      updateTriggers: { getLineColor: key },
    }),
    new deck.ScatterplotLayer({
      id: "found", data: hi, getPosition: pos, getFillColor: (i) => [...pointStyle(i)[0].slice(0, 3), 255], getRadius: 5,
      radiusUnits: "pixels", stroked: true, getLineColor: ink, lineWidthUnits: "pixels", getLineWidth: 1.6, pickable: true,
      updateTriggers: { getFillColor: key, getLineColor: key },
    }),
    new deck.ScatterplotLayer({
      id: "selected", data: S.selected !== null ? [S.selected] : [], getPosition: pos, getRadius: 9.5, radiusUnits: "pixels",
      filled: false, stroked: true, getLineColor: ink, lineWidthUnits: "pixels", getLineWidth: 2.5,
      updateTriggers: { getLineColor: key },
    }),
  ];
}
function labelLayer() {
  const key = [S.focus, C.dark, MAP.labels.length].join("|");
  return new deck.TextLayer({
    id: "labels", data: placedLabels(), getPosition: (l) => l.pos, getText: (l) => l.text,
    getSize: S.focus !== null ? 13 : 12, sizeUnits: "pixels", fontFamily: MONO, fontWeight: 600, characterSet: "auto",
    getColor: [...rgb(C.ink), 255], getTextAnchor: "middle", getAlignmentBaseline: "center",
    background: true, getBackgroundColor: [...rgb(C.surface), 235], getBorderColor: (l) => [...rgb(l.color), 255],
    getBorderWidth: 2, backgroundPadding: [8, 4], fontSettings: { sdf: true, fontSize: 64, buffer: 6 },
    updateTriggers: { getColor: key, getBackgroundColor: key, getBorderColor: key },
  });
}
function render() { if (MAP.deck) MAP.deck.setProps({ layers: [...MAP.base, labelLayer()] }); }
let labelFrame = 0;
function scheduleLabels() { if (!labelFrame) labelFrame = requestAnimationFrame(() => { labelFrame = 0; render(); }); }

function drawMap() {
  readTheme();
  computeOrder();
  computeLabels();
  MAP.base = baseLayers();
  $("#map-count").textContent = `${fmt(MAP.idx.length)} of ${fmt(N)} prompts shown`;
  render();
}
function createMap() {
  MAP.view = fitView();
  MAP.deck = new deck.Deck({
    parent: MAP.el,
    views: new deck.OrthographicView({ id: "map", flipY: false }),
    initialViewState: MAP.view,
    controller: { scrollZoom: { smooth: true, speed: 0.012 }, inertia: 380, doubleClickZoom: true, keyboard: true, dragRotate: false },
    onViewStateChange: ({ viewState }) => { MAP.view = viewState; hideTip(); scheduleLabels(); },
    getCursor: ({ isDragging, isHovering }) => (isDragging ? "grabbing" : isHovering ? "pointer" : "grab"),
    onHover: (info, ev) => {
      if (info.object != null && info.layer && info.layer.id !== "labels" && !ev?.leftButton) showTip(info.object, ev.srcEvent);
      else hideTip();
    },
    onClick: (info) => {
      if (info.object == null || !info.layer || info.layer.id === "labels") return;
      hideTip(); select(info.object); openDetail(info.object);
    },
    layers: [],
  });
}

function legend() {
  const ul = $("#legend");
  ul.innerHTML = "";
  D.groups.forEach((name, g) => {
    if (!D.groupCount[g]) return;
    const li = document.createElement("li");
    li.className = (S.focus === g ? "focus " : "") + (S.hidden.has(g) ? "off" : "");
    const sw = S.colorBy === "year" ? "transparent" : groupColor(g);
    li.innerHTML = `<span class="sw" style="background:${sw};${S.colorBy === "year" ? "box-shadow:inset 0 0 0 1.5px " + C.line : ""}"></span>
      <span class="name" title="${esc(name)}">${esc(name)}</span><span class="cnt">${fmt(D.groupCount[g])}</span>
      <button class="eye" title="${S.hidden.has(g) ? "Show" : "Hide"} ${esc(name)}" aria-label="toggle">${S.hidden.has(g) ? eyeOff : eyeOn}</button>`;
    li.addEventListener("click", (ev) => {
      if (ev.target.closest(".eye")) { S.hidden.has(g) ? S.hidden.delete(g) : S.hidden.add(g); if (S.focus === g) S.focus = null; }
      else if (g === UNASSIGNED || S.colorBy === "year") { S.hidden.has(g) ? S.hidden.delete(g) : S.hidden.add(g); }
      else { S.focus = S.focus === g ? null : g; S.hidden.delete(g); S.hiddenTypes.clear(); }
      refreshMap();
    });
    ul.appendChild(li);
  });
  // task types of the focused group
  const tb = $("#types-block");
  tb.hidden = S.focus === null || S.colorBy === "year";
  if (!tb.hidden) {
    $("#types-title").textContent = D.groups[S.focus];
    const tl = $("#type-legend");
    tl.className = "legend types";
    tl.innerHTML = "";
    D.groupTypes[S.focus].forEach((t) => {
      const slot = typeSlot(t);
      const li = document.createElement("li");
      li.className = S.hiddenTypes.has(t) ? "off" : "";
      li.innerHTML = `<span class="sw" style="background:${C.t[slot]}"></span>
        <span class="name" title="${esc(D.types[t])}">${esc(D.types[t])}</span><span class="cnt">${fmt(D.typeCount[t])}</span>`;
      li.addEventListener("click", () => { S.hiddenTypes.has(t) ? S.hiddenTypes.delete(t) : S.hiddenTypes.add(t); refreshMap(); });
      tl.appendChild(li);
    });
  }
  // year chips
  const yc = $("#year-chips");
  yc.innerHTML = "";
  D.years.forEach((y) => {
    const b = document.createElement("button");
    b.className = "chip" + (S.years.has(y) ? " on" : "");
    const n = P.year.reduce((s, v) => s + (v === y), 0);
    b.innerHTML = (S.colorBy === "year" ? `<span class="sw" style="background:${yearColor(y)}"></span>` : "") + y
      + (S.colorBy === "year" ? ` <span class="muted">${fmt(n)}</span>` : "");
    b.title = `${fmt(n)} prompts from papers first submitted in ${y}`;
    b.addEventListener("click", () => {
      if (S.years.has(y) && S.years.size > 1) S.years.delete(y); else S.years.add(y);
      refreshMap();
    });
    yc.appendChild(b);
  });
}
const eyeOn = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></svg>`;
const eyeOff = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3l18 18M10.6 5.1A10.8 10.8 0 0 1 12 5c6.4 0 10 7 10 7a17 17 0 0 1-3.2 4.2M6.6 6.6C3.9 8.4 2 12 2 12s3.6 7 10 7a9.8 9.8 0 0 0 5.4-1.6"/><path d="M9.9 9.9a3 3 0 0 0 4.2 4.2"/></svg>`;

function refreshMap() { readTheme(); legend(); drawMap(); }
function select(i) { S.selected = i; drawMap(); }
function focusPoint(i) {
  setTab("map");
  S.selected = i;
  if (!visible(i)) { S.hidden.delete(P.g[i]); S.hiddenTypes.delete(P.t[i]); S.years.add(P.year[i]); if (S.releasedOnly && !P.r[i]) { S.releasedOnly = false; $("#released-only").checked = false; } }
  refreshMap();
  const fit = MAP.fit || fitView();
  // center the prompt in the part of the map the detail drawer leaves visible
  const zoom = fit.zoom + 2.4, drawer = Math.min(460, MAP.el.clientWidth * 0.45);
  flyTo({ ...fit, target: [P.x[i] + drawer / 2 / 2 ** zoom, P.y[i], 0], zoom }, 900);
  openDetail(i);
}

/* ---------- tooltip ---------- */
function tipHtml(i) {
  const pa = D.papers[P.p[i]];
  const g = P.g[i];
  const color = S.colorBy === "year" ? yearColor(P.year[i]) : (S.focus === g ? C.t[typeSlot(P.t[i])] : groupColor(g));
  const sn = D.snippet[i];
  const text = sn ? `<p class="tt-text">${esc(sn.length > 180 ? sn.slice(0, 180) + "…" : sn)}</p>`
    : `<p class="tt-text na">Text not redistributed (arXiv license). Open the paper to read it.</p>`;
  const cluster = D.clusters[P.c[i]];
  return `<div class="tt-kicker"><span class="sw" style="background:${color}"></span>${esc(D.types[P.t[i]])}</div>
    <div class="tt-cluster">${g === UNASSIGNED ? "Not assigned to a task type" + suggestText(i, " · ") : esc(cluster)}</div>${text}
    <div class="tt-paper">${esc(pa.title)} · ${pa.year}${P.f[i] > 1 ? ` · family in ${P.f[i]} papers` : ""}</div>`;
}
function showTip(i, ev) {
  const tt = $("#tooltip");
  tt.innerHTML = tipHtml(i);
  tt.hidden = false;
  const pad = 14, w = tt.offsetWidth, h = tt.offsetHeight;
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + w > innerWidth - 8) x = ev.clientX - w - pad;
  if (y + h > innerHeight - 8) y = ev.clientY - h - pad;
  tt.style.left = Math.max(8, x) + "px"; tt.style.top = Math.max(8, y) + "px";
}
function hideTip() { $("#tooltip").hidden = true; }

/* ---------- detail ---------- */
async function openDetail(i) {
  // keep the drawer clear of the results list's scrollbar, whatever width the system draws it at
  const sm = $(".search-main");
  if (sm && sm.offsetWidth) document.documentElement.style.setProperty("--results-scrollbar", (sm.offsetWidth - sm.clientWidth) + "px");
  const pa = D.papers[P.p[i]];
  const g = P.g[i];
  const box = $("#detail"), body = $("#detail-body");
  box.hidden = false;
  pinCard(document.body.dataset.tab === "search" ? i : null);
  const back = document.body.dataset.tab === "search" ? `<button class="d-back" id="d-back">← Back To The Map</button>` : "";
  body.innerHTML = `${back}<div class="d-kicker"><span class="sw" style="background:${groupColor(g)}"></span>${esc(D.groups[g])}</div>
    <div class="d-type">${esc(D.types[P.t[i]])}</div>
    <div class="d-cluster">${g === UNASSIGNED ? "Not assigned to a task type by the clustering" : "Cluster: " + esc(D.clusters[P.c[i]])}</div>
    ${suggestBox(i)}
    <div class="d-paper"><a href="${arxivUrl(pa)}" target="_blank" rel="noopener">${esc(pa.title)}</a>
      <div class="meta">${esc(authorsShort(pa))} · ${pa.year} · arXiv:${pa.id}v${pa.v}</div></div>
    <div class="badges" id="d-badges"><span class="badge">${fmt(P.w[i])} words</span>
      ${P.f[i] > 1 ? `<span class="badge strong">Reused: family appears in ${P.f[i]} papers</span>` : ""}</div>
    <div id="d-text"><span class="spinner"></span></div>
    <div class="d-actions">
      <button class="btn primary" id="d-similar" title="Find prompts similar to this one">Find Similar</button>
      <button class="btn" id="d-map">Show On Map</button>
      ${P.r[i] ? `<button class="btn" id="d-copy">Copy Text</button>` : ""}
    </div>
    <p class="hint" style="margin-top:12px">Prompt ID ${esc(D.ids[i])}</p>`;
  $("#d-similar").onclick = () => runSimilar(i);
  $("#d-back")?.addEventListener("click", closeDetail);
  $("#d-map").onclick = () => focusPoint(i);
  const rec = await fetch(`/api/prompt?i=${i}`).then((r) => r.json());
  $("#d-badges").insertAdjacentHTML("afterbegin", `<span class="badge">${esc(licenseShort(rec.license))}</span>${rec.second_pass ? `<span class="badge">assigned in second pass</span>` : ""}`);
  const dt = $("#d-text");
  if (rec.text) {
    let raw = false;
    const render = () => {
      dt.innerHTML = `<div class="d-text-head"><h3 style="margin:0">Prompt</h3><button class="link" id="d-raw">${raw ? "Show Plain Text" : "Show LaTeX Source"}</button></div>
        <pre class="d-text">${esc(raw ? rec.text : rec.text_normalised || rec.text)}</pre>`;
      $("#d-raw").onclick = () => { raw = !raw; render(); };
    };
    render();
    const cp = $("#d-copy");
    if (cp) cp.onclick = () => navigator.clipboard.writeText(rec.text_normalised || rec.text).then(() => toast("Prompt copied"));
  } else {
    dt.innerHTML = `<div class="d-na">The paper is under the default arXiv license, so the prompt text is not redistributed.
      <a href="${arxivUrl(pa)}" target="_blank" rel="noopener">Open the paper</a> to read it, or re-extract it with the dataset's script.
      It is still part of the map and the search.</div>`;
  }
}
function closeDetail() {
  $("#detail").hidden = true;
  pinCard(null);
  if (MINI.hl !== null) drawMini(null);  // and the enlarged dot on the search map
  if (S.selected !== null) { S.selected = null; drawMap(); }  // drop the ring around the clicked prompt
}

/* ---------- search ---------- */
const MINI = { el: null, deck: null, hl: null, all: null, fitted: false };
function filters() {
  const g = $("#s-group").value, y = $("#s-year").value;
  return { groups: g, years: y, released: $("#s-released").checked ? 1 : 0 };
}
async function runSearch(text) {
  const q = (text ?? $("#q").value).trim();
  if (!q) { $("#q").focus(); return; }
  const btn = $("#go");
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> Searching`;
  try {
    const mode = S.model === "ready" ? S.mode : "keyword";
    const res = await fetch("/api/search", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ q, k: S.k, mode, ...filters() }),
    }).then((r) => r.json());
    if (res.error) throw new Error(res.error);
    S.lastSearch = { kind: "query", q, res };
    saveSearch({ kind: "query", q, mode: S.mode });
    renderResults();
  } catch (e) {
    $("#results-head").innerHTML = `<span>Search failed: ${esc(e.message)}</span>`;
  } finally {
    btn.disabled = false; btn.innerHTML = `Search <kbd>${KBD}</kbd>`;
  }
}
async function runSimilar(i) {
  closeDetail();
  setTab("search");
  const p = new URLSearchParams({ i, k: S.k, ...filters() });
  const [res, rec] = await Promise.all([
    fetch(`/api/similar?${p}`).then((r) => r.json()),
    P.r[i] ? fetch(`/api/prompt?i=${i}`).then((r) => r.json()).catch(() => null) : null,
  ]);
  // the box shows the prompt we search from; prompts under the default arXiv license are not shown, so it is emptied
  $("#q").value = rec ? rec.text_normalised || rec.text || "" : "";
  $("#q").scrollTop = 0;
  $("#q").dispatchEvent(new Event("input"));
  S.lastSearch = { kind: "similar", of: i, res };
  saveSearch({ kind: "similar", of: i });
  renderResults();
}
function saveSearch(s) {
  try {
    if (s) sessionStorage.setItem("sepia-search", JSON.stringify({ ...s, ...filters(), k: S.k }));
    else sessionStorage.removeItem("sepia-search");
  } catch (e) { /* storage blocked: nothing to restore after a reload */ }
}
function restoreSearch() {
  let s = null;
  try { s = JSON.parse(sessionStorage.getItem("sepia-search") || "null"); } catch (e) { s = null; }
  if (!s || typeof s !== "object") return;
  if (RESULT_SIZES.includes(+s.k)) S.k = +s.k;
  $("#s-group").value = s.groups || ""; $("#s-year").value = s.years || ""; $("#s-released").checked = String(s.released) === "1";
  if (s.kind === "similar" && Number.isInteger(s.of) && s.of >= 0 && s.of < N) { runSimilar(s.of); return; }
  if (s.kind !== "query" || typeof s.q !== "string" || !s.q) return;
  $("#q").value = s.q;
  $("#q").dispatchEvent(new Event("input"));
  if (s.mode === "keyword" || s.mode === "semantic") setMode(s.mode);
  // a semantic query has to wait for the embedding model, which loads a few seconds after the page
  if (S.mode === "semantic" && S.model !== "ready" && S.model !== "off") {
    S.pendingRestore = s.q;
    $("#results-head").innerHTML = `<div class="rh-main"><span class="spinner"></span> Restoring your last search once the embedding model is ready…</div>`;
    $("#empty").hidden = true;
  } else runSearch(s.q);
}
function resetSearch() {
  // back to the empty search page: no results, no breakdown, the overview map
  S.lastSearch = null;
  S.pendingRestore = null;
  saveSearch(null);
  closeDetail();
  $("#results").innerHTML = "";
  $("#results-head").innerHTML = "";
  $("#empty").hidden = false;
  $("#breakdown-card").hidden = true;
  $("#results-foot").hidden = true;
  drawMini(null);
  fitMini(600);
}
function renderResults() {
  const { res } = S.lastSearch;
  $("#empty").hidden = true;
  $("#results-foot").hidden = false;
  const head = $("#results-head");
  if (S.lastSearch.kind === "similar") {
    const i = S.lastSearch.of, pa = D.papers[P.p[i]];
    head.innerHTML = `<div class="rh-main">Prompts closest to the <strong>${esc(D.types[P.t[i]])}</strong> prompt from <a href="#" id="of-link">${esc(pa.title)}</a></div>
      <div class="rh-meta">embedding neighbors · ${kSelect()}</div>`;
    $("#of-link").onclick = (e) => { e.preventDefault(); openDetail(i); };
  } else {
    const fallback = res.mode === "keyword" && S.model !== "ready" && S.mode === "semantic" ? " · model not ready, used keywords" : "";
    head.innerHTML = `<div class="rh-main"><strong>${res.results.length}</strong> prompts ${res.mode === "semantic" ? "closest in meaning to" : "matching the words of"} your query</div>
      <div class="rh-meta">${res.mode} · ${res.took_ms} ms${fallback} · ${kSelect()}</div>`;
  }
  const ol = $("#results");
  ol.innerHTML = "";
  if (!res.results.length) {
    ol.innerHTML = `<li class="no-results">No prompts match. Try other words, semantic mode, or fewer filters.</li>`;
  }
  const max = res.results.length ? res.results[0][1] : 1;
  res.results.forEach(([i, score, sameFam], k) => {
    const pa = D.papers[P.p[i]];
    const g = P.g[i];
    const li = document.createElement("li");
    li.className = "card";
    li.dataset.i = i;
    li.style.setProperty("--gc", groupColor(g));
    const sn = D.snippet[i];
    const pct = res.mode === "keyword" ? Math.round((score / max) * 100) : Math.round(Math.max(0, Math.min(1, (score - 0.3) / 0.7)) * 100);
    const sub = g === UNASSIGNED ? `<span class="c-suggest">${suggestText(i, "")}</span>` : esc(D.clusters[P.c[i]]);
    const badge = sameFam ? `<span class="badge strong">near-duplicate</span>` : P.f[i] > 1 ? `<span class="badge">family in ${P.f[i]} papers</span>` : "";
    li.innerHTML = `<div class="c-head">
        <span class="rank">${k + 1}</span>
        <span class="c-type"><span class="sw"></span>${esc(D.types[P.t[i]])}</span>
        <span class="c-cluster">${sub}</span>
        <span class="score" title="${res.mode === "keyword" ? "TF-IDF score" : "cosine similarity"}"><span class="bar"><i style="width:${pct}%"></i></span><b>${score.toFixed(res.mode === "keyword" ? 3 : 2)}</b></span>
      </div>
      <a class="c-title" href="${arxivUrl(pa)}" target="_blank" rel="noopener">${esc(pa.title)}</a>
      <div class="c-meta">${esc(authorsShort(pa))} · ${pa.year} · arXiv:${pa.id}</div>
      ${sn ? `<div class="c-text">${esc(sn)}</div>` : `<div class="c-text na">Text not redistributed (arXiv license). Open the paper to read it.</div>`}
      <div class="c-actions">
        ${P.r[i] ? `<button class="act" data-act="full">Full Prompt</button>` : ""}
        <button class="act" data-act="detail">Details</button>
        <button class="act" data-act="similar">More Like This</button>
        <button class="act" data-act="map">Show On Map</button>
        ${badge}
      </div>`;
    li.addEventListener("click", async (e) => {
      const act = e.target.closest("[data-act]")?.dataset.act;
      if (!act) return;
      if (act === "detail") openDetail(i);
      if (act === "similar") runSimilar(i);
      if (act === "map") focusPoint(i);
      if (act === "full") {
        const box = $(".c-text", li);
        if (box.classList.contains("full")) { box.classList.remove("full"); box.textContent = sn; e.target.textContent = "Full Prompt"; return; }
        const rec = await fetch(`/api/prompt?i=${i}`).then((r) => r.json());
        box.classList.add("full"); box.textContent = rec.text_normalised || rec.text; e.target.textContent = "Collapse";
      }
    });
    li.addEventListener("mouseenter", () => highlightMini(i));
    li.addEventListener("mouseleave", () => highlightMini(null));
    ol.appendChild(li);
  });
  // more results, one size up, when the list is full
  const next = RESULT_SIZES.find((n) => n > S.k);
  if (next && res.results.length >= S.k) {
    const more = document.createElement("li");
    more.className = "more-row";
    more.innerHTML = `<button class="act more" type="button">Show ${next} Results</button>`;
    more.querySelector("button").onclick = () => setK(next);
    ol.appendChild(more);
  }
  $("#k-select")?.addEventListener("change", (e) => setK(+e.target.value));
  if (S.pinned != null && !$("#detail").hidden) pinCard(S.pinned, false);
  drawMini(null);
  fitMini();
  breakdown();
}
/* the result whose detail panel is open stays highlighted (and in view) until the panel closes */
function pinCard(i, scroll = true) {
  $$(".results .card.pinned").forEach((c) => c.classList.remove("pinned"));
  S.pinned = i;
  if (i === null) return;
  const c = $(`.results .card[data-i="${i}"]`);
  if (!c) return;
  c.classList.add("pinned");
  if (scroll) revealCard(c, true);
}
/* bring a result card into view in the results column: centered when pinned, just enough when hovered on the map */
function revealCard(c, center) {
  const sm = $(".search-main");
  if (!sm) return;
  const r = c.getBoundingClientRect(), v = sm.getBoundingClientRect(), pad = 16;
  let y = null;
  const inView = r.top >= v.top + pad && r.bottom <= v.bottom - pad;
  if (center && !inView) y = sm.scrollTop + (r.top - v.top) - Math.max(pad, (v.height - r.height) / 2);
  else if (center) return;
  else if (r.top < v.top + pad) y = sm.scrollTop - (v.top + pad - r.top);
  else if (r.bottom > v.bottom - pad) y = sm.scrollTop + (r.bottom - (v.bottom - pad));
  if (y === null) return;
  if (sm.smoothTo) sm.smoothTo(y); else sm.scrollTop = y;
}
function kSelect() {
  return `<label class="k-pick">show <select id="k-select" class="pill-select" aria-label="Number of results">${
    RESULT_SIZES.map((n) => `<option value="${n}"${n === S.k ? " selected" : ""}>${n}</option>`).join("")}</select></label>`;
}
function setK(n) {
  if (!RESULT_SIZES.includes(n) || !S.lastSearch) return;
  S.k = n;
  S.lastSearch.kind === "similar" ? runSimilar(S.lastSearch.of) : runSearch(S.lastSearch.q);
}

/* search map: same deck.gl setup as the cluster map; only a new result set moves the view */
function miniFit() {
  const res = S.lastSearch?.res;
  const ids = res ? res.results.map((r) => r[0]) : [];
  const place = res?.place;
  const { x0, x1, y0, y1 } = extent();
  const w = MINI.el.clientWidth || 520, h = MINI.el.clientHeight || 420;
  let b = [x0, x1, y0, y1];
  if (ids.length) {
    // frame the closest results and the query; a few far-off lower-ranked hits should not zoom the view out
    const core = ids.slice(0, 12);
    const xs = core.map((i) => P.x[i]).concat(place ? [place[0]] : []);
    const ys = core.map((i) => P.y[i]).concat(place ? [place[1]] : []);
    const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
    const pad = Math.max(0.55, 0.3 * span);
    b = [Math.min(...xs) - pad, Math.max(...xs) + pad, Math.min(...ys) - pad, Math.max(...ys) + pad];
  }
  const zoomFor = (bb) => Math.log2(Math.min(w / ((bb[1] - bb[0]) * 1.04), h / ((bb[3] - bb[2]) * 1.04)));
  const full = zoomFor([x0, x1, y0, y1]);
  return { target: [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, 0], zoom: zoomFor(b), minZoom: full - 1, maxZoom: full + 7 };
}
function fitMini(ms = 700) {
  if (!MINI.deck) return;
  MINI.deck.setProps({
    initialViewState: { ...miniFit(), transitionDuration: ms, transitionEasing: easeInOut,
      transitionInterpolator: new deck.LinearInterpolator(["target", "zoom"]) },
  });
  MINI.fitted = true;
}
function createMini() {
  MINI.all = Array.from({ length: N }, (_, i) => i);
  MINI.deck = new deck.Deck({
    parent: MINI.el,
    views: new deck.OrthographicView({ id: "mini", flipY: false }),
    initialViewState: miniFit(),
    controller: { scrollZoom: { smooth: true, speed: 0.012 }, inertia: 380, doubleClickZoom: true, keyboard: false, dragRotate: false },
    onViewStateChange: () => hideTip(),
    getCursor: ({ isDragging, isHovering }) => (isDragging ? "grabbing" : isHovering ? "pointer" : "grab"),
    onHover: (info, ev) => {
      if (info.object != null && info.layer?.id === "mini-hits" && !ev?.leftButton) { showTip(info.object, ev.srcEvent); markCard(info.object); }
      else { hideTip(); markCard(null); }
    },
    onClick: (info) => { if (info.object != null && info.layer?.id === "mini-hits") { hideTip(); drawMini(info.object); openDetail(info.object); } },
    layers: [],
  });
}
function drawMini(hl = MINI.hl) {
  MINI.hl = hl;
  if (!MINI.deck) return;
  readTheme();
  const res = S.lastSearch?.res;
  const ids = res ? res.results.map((r) => r[0]) : [];
  const place = res?.place;
  const key = [C.dark, C.surface].join("|");
  const pos = (i) => [P.x[i], P.y[i]];
  const ink = [...rgb(C.ink), 255], page = rgb(C.page);
  MINI.deck.setProps({
    layers: [
      new deck.ScatterplotLayer({
        id: "mini-base", data: MINI.all, getPosition: pos, getFillColor: [...rgb(C.muted), C.dark ? 84 : 70], getRadius: 1.9,
        radiusUnits: "pixels", updateTriggers: { getFillColor: key },
      }),
      new deck.ScatterplotLayer({
        id: "mini-hits", data: ids, getPosition: pos, getFillColor: (i) => [...rgb(groupColor(P.g[i])), 255],
        getRadius: (i) => (i === hl ? 8.5 : 5.2), radiusUnits: "pixels", stroked: true, getLineColor: [...page, 255],
        lineWidthUnits: "pixels", getLineWidth: 1.5, pickable: true, autoHighlight: true, highlightColor: [...rgb(C.ink), 110],
        transitions: { getRadius: 140 }, updateTriggers: { getFillColor: key, getLineColor: key, getRadius: hl },
      }),
      // where the query falls (or the prompt a "more like this" started from): a ring around a dot
      new deck.ScatterplotLayer({
        id: "mini-query", data: place ? [place] : [], getPosition: (d) => d, getRadius: 10, radiusUnits: "pixels",
        getFillColor: [...page, 150], stroked: true, getLineColor: ink, lineWidthUnits: "pixels", getLineWidth: 2.5,
        updateTriggers: { getFillColor: key, getLineColor: key },
      }),
      new deck.ScatterplotLayer({
        id: "mini-query-dot", data: place ? [place] : [], getPosition: (d) => d, getRadius: 3, radiusUnits: "pixels",
        getFillColor: ink, updateTriggers: { getFillColor: key },
      }),
      new deck.TextLayer({
        id: "mini-ranks", data: ids.slice(0, 5).map((i, k) => ({ i, k })), getPosition: (d) => pos(d.i), getText: (d) => String(d.k + 1),
        getSize: 11, sizeUnits: "pixels", getPixelOffset: [0, -14], fontFamily: MONO, fontWeight: 700, getColor: ink,
        fontSettings: { sdf: true, fontSize: 64, buffer: 8 }, outlineWidth: 3, outlineColor: [...page, 255],
        updateTriggers: { getColor: key },
      }),
    ],
  });
}
function highlightMini(i) {
  const v = i ?? S.pinned ?? null;  // with a panel open, the pinned prompt stays marked when the pointer moves on
  if (v !== MINI.hl) drawMini(v);
}
function markCard(i) {
  $$(".card.hl").forEach((c) => c.classList.remove("hl"));
  if (i === null) return;
  const c = $(`.card[data-i="${i}"]`);
  if (c) { c.classList.add("hl"); revealCard(c, false); }
}
function breakdown() {
  const res = S.lastSearch.res;
  const cnt = new Map();
  res.results.forEach(([i]) => cnt.set(P.t[i], (cnt.get(P.t[i]) || 0) + 1));
  const rows = [...cnt.entries()].sort((a, b) => b[1] - a[1]);
  const max = rows.length ? rows[0][1] : 1;
  $("#breakdown-card").hidden = !rows.length;
  $("#breakdown").innerHTML = rows.map(([t, n]) => {
    const g = D.typeGroup[t];
    return `<div class="bd-row"><span class="nm" title="${esc(D.types[t])}"><span class="sw" style="background:${groupColor(g)}"></span>${esc(D.types[t])}</span>
      <span class="track"><i style="width:${(n / max) * 100}%;background:${groupColor(g)}"></i></span><span class="v">${n}</span></div>`;
  }).join("");
}

/* ---------- status ---------- */
async function pollStatus() {
  try {
    const s = await fetch("/api/status").then((r) => r.json());
    const pill = $("#model-pill");
    if (s.model === "ready") {
      S.model = "ready";
      pill.className = "pill ready"; pill.textContent = "Semantic search ready";
      pill.title = s.check ? `Query embeddings match the dataset (self-check cosine ${s.check.toFixed(3)})` : "";
      $('#mode [data-v="semantic"]').disabled = false;
      if (S.pendingRestore) { const q = S.pendingRestore; S.pendingRestore = null; runSearch(q); }
      return;
    }
    if (s.model === "loading" || s.model === "not loaded") {
      S.model = "loading";
      pill.className = "pill loading"; pill.textContent = "Loading embedding model…";
      setTimeout(pollStatus, 2000);
    } else {
      S.model = "off";
      pill.className = "pill"; pill.textContent = "Keyword search only"; pill.title = s.model;
      setMode("keyword");
      $('#mode [data-v="semantic"]').disabled = true;
      if (S.pendingRestore) { const q = S.pendingRestore; S.pendingRestore = null; runSearch(q); }
    }
  } catch { setTimeout(pollStatus, 4000); }
}

/* ---------- ui ---------- */
const SUN = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';
const MOON = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>';
function setTheme(theme, redraw = true) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem("sepia-theme", theme); } catch (e) { /* storage blocked: keep it for this visit */ }
  const next = theme === "dark" ? "light" : "dark";
  const btn = $("#theme-toggle");
  btn.innerHTML = theme === "dark" ? SUN : MOON;
  btn.title = `Switch to ${next} mode`;
  btn.setAttribute("aria-label", btn.title);
  if (redraw) {
    refreshMap(); drawMini();
    // recolor the current results in place (re-rendering them would also refit the search map)
    $$("#results .card").forEach((c) => c.style.setProperty("--gc", groupColor(P.g[+c.dataset.i])));
    if (S.lastSearch) breakdown();
  }
}
const PANEL_HIDE = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M9 4v16M16 10l-2 2 2 2"/></svg>';
const PANEL_SHOW = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M9 4v16M14 10l2 2-2 2"/></svg>';
function setSide(open, animate = true) {
  // was the map showing the whole dataset? then refit it to the new width once the panel has moved
  const v = MAP.view, f = MAP.fit;
  const wasFit = v && f && Math.abs(v.zoom - f.zoom) < 0.02 && Math.hypot(v.target[0] - f.target[0], v.target[1] - f.target[1]) < 0.05;
  S.sideOpen = open;
  $("#tab-map").classList.toggle("collapsed", !open);
  const btn = $("#side-toggle");
  btn.innerHTML = open ? PANEL_HIDE : PANEL_SHOW;
  btn.title = open ? "Hide the side panel" : "Show the side panel";
  btn.setAttribute("aria-label", btn.title);
  btn.setAttribute("aria-expanded", String(open));
  try { localStorage.setItem("sepia-side", open ? "open" : "closed"); } catch (e) { /* storage blocked */ }
  if (!animate) return;
  setTimeout(() => { if (wasFit) flyTo(fitView(), 420); else scheduleLabels(); }, 300);
}
function setNote(show, animate = true) {
  // the "Reading the map" card can be closed for a clear view; a small pill brings it back
  const v = MAP.view, f = MAP.fit;
  const wasFit = v && f && Math.abs(v.zoom - f.zoom) < 0.02 && Math.hypot(v.target[0] - f.target[0], v.target[1] - f.target[1]) < 0.05;
  $("#map-note").hidden = !show;
  $("#mn-open").hidden = show;
  try { localStorage.setItem("sepia-note", show ? "shown" : "hidden"); } catch (e) { /* storage blocked */ }
  if (animate && wasFit) flyTo(fitView(), 420);
}
/* smooth scrolling: wheel and trackpad steps ease toward their target instead of jumping */
function smoothWheel(el) {
  let target = el.scrollTop, raf = 0;
  const max = () => el.scrollHeight - el.clientHeight;
  const step = () => {
    const cur = el.scrollTop, next = cur + (target - cur) * 0.16;
    el.scrollTop = Math.abs(target - next) < 0.5 ? target : next;
    raf = Math.abs(target - el.scrollTop) >= 0.5 ? requestAnimationFrame(step) : 0;
  };
  el.addEventListener("wheel", (e) => {
    if (e.ctrlKey || e.defaultPrevented || Math.abs(e.deltaX) > Math.abs(e.deltaY)) return;
    // a box inside that scrolls on its own (the search box, an expanded prompt) keeps the wheel while it can
    const inner = e.target.closest("textarea, select, .c-text.full, .d-text, .tooltip");
    if (inner && inner !== el && inner.scrollHeight > inner.clientHeight) return;
    if (max() <= 0) return;
    const d = e.deltaMode === 1 ? e.deltaY * 18 : e.deltaMode === 2 ? e.deltaY * el.clientHeight : e.deltaY;
    e.preventDefault();
    if (!raf) target = el.scrollTop;
    target = Math.max(0, Math.min(max(), target + d));
    if (!raf) raf = requestAnimationFrame(step);
  }, { passive: false });
  // scrollbar drags, keys and scrollIntoView move the page too; start from there next time
  el.addEventListener("scroll", () => { if (!raf) target = el.scrollTop; }, { passive: true });
  // the same easing for scrolls the app starts itself (the browser's smooth scrollIntoView can stall here)
  el.smoothTo = (y) => {
    if (!raf) target = el.scrollTop;
    target = Math.max(0, Math.min(max(), y));
    if (!raf) raf = requestAnimationFrame(step);
  };
}
function syncHeader() { document.documentElement.style.setProperty("--header-h", $(".top").offsetHeight + "px"); }
function resetAll() {
  // back to the starting view: the full map, colored by activity, nothing filtered or selected
  closeDetail();
  Object.assign(S, { colorBy: "group", focus: null, releasedOnly: false, find: "", findSet: null, selected: null, labels: true });
  S.hidden.clear(); S.hiddenTypes.clear(); S.years = new Set(D.years);
  $$("#color-by button").forEach((x) => x.classList.toggle("on", x.dataset.v === "group"));
  $("#released-only").checked = false; $("#show-labels").checked = true; $("#map-find").value = "";
  S.suggest = false; $("#show-suggest").checked = false;
  setTab("map");
  refreshMap();
  flyTo(fitView(), 650);
}
function setTab(t) {
  $$(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === t)));
  $$(".tab").forEach((s) => s.classList.toggle("active", s.id === `tab-${t}`));
  document.body.dataset.tab = t;
  if (t === "about") closeDetail();  // the prompt panel belongs to the map and the search
  try { history.replaceState(null, "", "#" + t); } catch (e) { /* file:// or sandboxed */ }
  hideTip();
  requestAnimationFrame(() => {
    MAP.deck?.redraw(true);
    MINI.deck?.redraw(true);
    if (t === "search" && !MINI.fitted) fitMini(0);  // first visit: size the overview to the now-visible panel
  });
}
function setMode(m) { S.mode = m; $$("#mode button").forEach((b) => b.classList.toggle("on", b.dataset.v === m)); }
function toast(msg) { const t = $("#toast"); t.textContent = msg; t.hidden = false; clearTimeout(toast.t); toast.t = setTimeout(() => (t.hidden = true), 1600); }

function wire() {
  $$(".tabs button").forEach((b) => b.addEventListener("click", () => setTab(b.dataset.tab)));
  $$("#color-by button").forEach((b) => b.addEventListener("click", () => {
    S.colorBy = b.dataset.v; $$("#color-by button").forEach((x) => x.classList.toggle("on", x === b)); refreshMap();
  }));
  $("#unfocus").onclick = () => { S.focus = null; S.hiddenTypes.clear(); refreshMap(); };
  $("#released-only").onchange = (e) => { S.releasedOnly = e.target.checked; refreshMap(); };
  $("#show-labels").onchange = (e) => { S.labels = e.target.checked; drawMap(); };
  $("#show-suggest").onchange = (e) => { S.suggest = e.target.checked; drawMap(); };
  let ft = null;
  $("#map-find").addEventListener("input", (e) => { clearTimeout(ft); ft = setTimeout(() => setFind(e.target.value), 200); });
  $("#reset-view").onclick = () => flyTo(fitView(), 650);
  $("#detail-close").onclick = closeDetail;
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDetail(); });
  // search form
  D.groups.forEach((g, k) => { if (D.groupCount[k]) $("#s-group").insertAdjacentHTML("beforeend", `<option value="${k}">${esc(g)}</option>`); });
  D.years.forEach((y) => $("#s-year").insertAdjacentHTML("beforeend", `<option value="${y}">${y}</option>`));
  const syncClear = () => { $("#q-clear").hidden = !$("#q").value; };
  $("#q").addEventListener("input", syncClear);
  $("#q-clear").onclick = () => { $("#q").value = ""; syncClear(); resetSearch(); $("#q").focus(); };
  $("#go").innerHTML = `Search <kbd>${KBD}</kbd>`;
  $("#search-form").addEventListener("submit", (e) => { e.preventDefault(); runSearch(); });
  $("#q").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); runSearch(); } });
  $$("#mode button").forEach((b) => b.addEventListener("click", () => { if (!b.disabled) setMode(b.dataset.v); }));
  ["#s-group", "#s-year", "#s-released"].forEach((s) => $(s).addEventListener("change", () => {
    if (!S.lastSearch) return;
    S.lastSearch.kind === "similar" ? runSimilar(S.lastSearch.of) : runSearch(S.lastSearch.q);
  }));
  const ex = $("#examples");
  EXAMPLES.forEach(([label, t]) => {
    const b = document.createElement("button");
    b.type = "button"; b.className = "ex"; b.textContent = label; b.title = t;
    b.onclick = () => { $("#q").value = t; syncClear(); runSearch(t); };
    ex.appendChild(b);
  });

  $("#theme-toggle").onclick = () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
  $("#home").onclick = resetAll;
  [".about-main", ".search-main", "#map-side", ".search-side", "#detail"].forEach((q) => { const el = $(q); if (el) smoothWheel(el); });
  $("#side-toggle").onclick = () => setSide(!S.sideOpen);
  $("#mn-close").onclick = () => setNote(false);
  $("#mn-open").onclick = () => setNote(true);
  let note = "shown";
  try { note = localStorage.getItem("sepia-note") || "shown"; } catch (e) { /* storage blocked */ }
  if (new URLSearchParams(location.search).get("note") === "0") note = "hidden";  // ?note=0: a clean map, e.g. for figures
  setNote(note !== "hidden", false);
  let side = "open";
  try { side = localStorage.getItem("sepia-side") || "open"; } catch (e) { /* storage blocked */ }
  setSide(side !== "closed", false);
  syncHeader();
  addEventListener("resize", syncHeader);
  setTheme(document.documentElement.dataset.theme || "dark", false);
}

(async function init() {
  MAP.el = $("#map");
  MINI.el = $("#mini");
  // the map labels are rasterized once, so wait (briefly) for the web font first
  await Promise.race([document.fonts.load('600 12px "JetBrains Mono"'), new Promise((r) => setTimeout(r, 2500))]).catch(() => {});
  await load();
  readTheme();
  wire();
  legend();
  createMap();
  drawMap();
  createMini();
  drawMini();
  pollStatus();
  // a link can carry a search: ?q=...&mode=keyword&k=50 opens Prompt Search and runs it
  const link = new URLSearchParams(location.search);
  if (link.get("q")) {
    try { sessionStorage.setItem("sepia-search", JSON.stringify({ kind: "query", q: link.get("q").slice(0, 4000),
      mode: link.get("mode") === "keyword" ? "keyword" : "semantic", k: +link.get("k") || 30 })); } catch (e) { /* storage blocked */ }
    setTab("search");
    restoreSearch();
  } else if (location.hash === "#search") { setTab("search"); restoreSearch(); }
  else if (location.hash === "#about") setTab("about");
})();
