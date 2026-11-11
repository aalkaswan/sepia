---
title: SEPIA Explorer
emoji: 🦑
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: Explore and search prompts from SE research papers
---

# SEPIA Explorer

SEPIA (Software Engineering Prompts In Articles) collects the prompts that software engineering researchers report in their papers.
This app has three views of the dataset ([on Hugging Face](https://huggingface.co/datasets/aalkaswan/sepia)):

- **Cluster Map**: every prompt as a dot, placed by a 2D UMAP projection of its embedding and colored by activity group (or year). Click a group to focus on its task types, hover for a preview, click for the full prompt, its paper and license.
- **Prompt Search**: paste a prompt or describe a task and get the most similar prompts from the dataset, with their task type, paper and position on the map. "Find Similar" finds the embedding neighbors of any prompt.
- **About Us**: the team, where to find the dataset, our plans and how to contact us.

Semantic search embeds the query the same way the dataset was embedded (LaTeX stripped, template slots masked, the task instruction, `microsoft/harrier-oss-v1-0.6b` at revision `f9b9dc8`). On start the server re-embeds a few dataset prompts and checks them against the stored vectors. Without the model, free-text search falls back to TF-IDF keyword matching.

Unassigned prompts get a **suggested task type** when at least 7 of their 10 nearest typed prompts (own family excluded) share one: 1,094 of the 3,073 unassigned prompts. On typed prompts the same vote agrees with the real label 99% of the time where it applies (84% of them). Suggestions are shown in tooltips, the detail panel and search results, and as rings on the map; they are not labels in the released dataset.

Prompts from papers under the default arXiv license are searchable and shown on the map, but their text is not served; the app links to the paper instead.

## Run locally

```bash
pip install torch==2.5.1 -r requirements.txt
python server.py --port 8765
```

Then open http://localhost:8765. The model (1.2 GB) is downloaded from Hugging Face on the first start and loads in about 10 s on a laptop CPU.
The `Dockerfile` is what the Hugging Face Space runs (port 7860, model baked into the image).

## Rebuild the data

`data/` is built from the released `prompts.jsonl` and `papers.csv` plus the prompt embeddings:

```bash
pip install umap-learn
python build_data.py --prompts prompts.jsonl --papers papers.csv \
    --emb prompt_embeddings_part1.npy prompt_embeddings_part2.npy --ids prompt_embeddings_ids.csv
```

## Files

- `server.py`: static files plus `/api/status`, `/api/prompt`, `/api/similar`, `/api/search`, `/api/find`, `/api/suggestions`
- `build_data.py`: UMAP projection and the data bundle
- `static/`: the page (`index.html`, `app.js`, `style.css`; deck.gl from jsDelivr, JetBrains Mono from Google Fonts), the logos and the team photos
- `data/`: `points.json` (map), `records.json` (prompt details), `embeddings.npy` (float16)
