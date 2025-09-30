# src/embed.py

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ─── Paths & Setup ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNK_DIR = os.path.join(BASE_DIR, "chunks")
INDEX_DIR = os.path.join(BASE_DIR, "indices")
os.makedirs(INDEX_DIR, exist_ok=True)

# ─── 1️⃣ Load text chunks ───────────────────────────────────────────────────
texts, text_meta = [], []
for fn in sorted(os.listdir(CHUNK_DIR)):
    if fn.endswith(".txt"):
        path = os.path.join(CHUNK_DIR, fn)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:  # Only add non-empty chunks
                texts.append(content)
                text_meta.append(fn)

if not texts:
    raise RuntimeError(f"No text chunks found in {CHUNK_DIR!r}")

# ─── 2️⃣ Load image captions ────────────────────────────────────────────────
image_meta, captions = [], []
for fn in os.listdir(CHUNK_DIR):
    if fn.endswith("_images.json"):
        try:
            meta = json.load(
                open(os.path.join(CHUNK_DIR, fn), encoding="utf-8"))
            for item in meta:
                if item.get("caption", "").strip():
                    image_meta.append(item)
                    captions.append(item["caption"])
        except Exception as e:
            print(f"⚠️ Error loading {fn}: {e}")

print(f"Loaded {len(texts)} text chunks, {len(captions)} image captions")

# ─── 3️⃣ Compute SBERT embeddings ───────────────────────────────────────────
model = SentenceTransformer("all-MiniLM-L6-v2")

# text embeddings
if texts:
    t_embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    t_embs = t_embs.astype("float32")
    faiss.normalize_L2(t_embs)
else:
    t_embs = np.zeros((0, 384), dtype="float32")  # Default dimension

# caption embeddings
if captions:
    c_embs = model.encode(captions, convert_to_numpy=True,
                          show_progress_bar=True)
    c_embs = c_embs.astype("float32")
    faiss.normalize_L2(c_embs)
else:
    c_embs = np.zeros((0, t_embs.shape[1] if len(
        t_embs) > 0 else 384), dtype="float32")

# ─── 4️⃣ Build & save FAISS indices ─────────────────────────────────────────
# text index
if len(t_embs) > 0:
    dim = t_embs.shape[1]
    text_index = faiss.IndexFlatIP(dim)
    text_index.add(t_embs)
    faiss.write_index(text_index, os.path.join(
        INDEX_DIR, "faiss_text_index.bin"))
    with open(os.path.join(INDEX_DIR, "faiss_text_meta.json"), "w", encoding="utf-8") as f:
        json.dump(text_meta, f, indent=2)
    print(f"✅ Text index: {len(text_meta)} vectors")

# caption index
if len(c_embs) > 0:
    dim_c = c_embs.shape[1]
    cap_index = faiss.IndexFlatIP(dim_c)
    cap_index.add(c_embs)
    faiss.write_index(cap_index, os.path.join(
        INDEX_DIR, "faiss_caption_index.bin"))
    with open(os.path.join(INDEX_DIR, "faiss_caption_meta.json"), "w", encoding="utf-8") as f:
        json.dump(image_meta, f, indent=2)
    print(f"✅ Caption index: {len(image_meta)} vectors")

print("Embedding process completed successfully!")
