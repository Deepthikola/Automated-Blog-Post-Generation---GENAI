# src/query.py

import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ─── Config & Paths ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(BASE_DIR, "indices")

T_INDEX_FILE = os.path.join(INDEX_DIR, "faiss_text_index.bin")
T_META_FILE = os.path.join(INDEX_DIR, "faiss_text_meta.json")
C_INDEX_FILE = os.path.join(INDEX_DIR, "faiss_caption_index.bin")
C_META_FILE = os.path.join(INDEX_DIR, "faiss_caption_meta.json")

TOP_K_TEXT = 5
TOP_K_IMG = 10
SIM_THRESH = 0.25  # Lowered threshold to get more images

# ─── Load models & indices ──────────────────────────────────────────────────
model = SentenceTransformer("all-MiniLM-L6-v2")

text_index = faiss.read_index(T_INDEX_FILE)
text_meta = json.load(open(T_META_FILE, encoding="utf-8"))

cap_index = faiss.read_index(C_INDEX_FILE)
cap_meta = json.load(open(C_META_FILE, encoding="utf-8"))


def retrieve(query: str):
    # embed & normalize
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = q_emb.astype("float32")
    faiss.normalize_L2(q_emb)

    results = {"texts": [], "images": []}

    # ─ Text retrieval ─────────────────────────────
    D_t, I_t = text_index.search(q_emb, TOP_K_TEXT)
    for score, idx in zip(D_t[0], I_t[0]):
        if score < SIM_THRESH:
            continue
        results["texts"].append({
            "chunk_id": text_meta[idx],
            "text": open(os.path.join(BASE_DIR, "chunks", text_meta[idx]), encoding="utf-8").read(),
            "score": float(score),
        })

    # ─ Image retrieval ────────────────────────────
    D_c, I_c = cap_index.search(q_emb, TOP_K_IMG)
    for score, idx in zip(D_c[0], I_c[0]):
        if score < SIM_THRESH:
            continue
        item = cap_meta[idx]
        # Convert absolute path to relative path for web compatibility
        rel_path = os.path.relpath(item["path"], BASE_DIR).replace("\\", "/")
        results["images"].append({
            "path": rel_path,
            "caption": item["caption"],
            "page": item["page"],
            "score": float(score),
        })

    return results


if __name__ == "__main__":
    q = input("🔍 Enter your query: ").strip()
    out = retrieve(q)

    print("\nTop Text Matches:")
    for t in out["texts"]:
        print(f"- ({t['score']:.4f}) {t['text'][:100]}...")

    print("\nRelevant Images:")
    for im in out["images"]:
        print(f"- ({im['score']:.2f}) {im['caption']} @ {im['path']}")
