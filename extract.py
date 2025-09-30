# src/extract.py (FIXED VERSION)

import os
import re
import json
import fitz
from PIL import Image
from transformers import pipeline

# ─── Config ──────────────────────────────────────────────────────────────────
CHUNK_SIZE = 500
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT_DIR = os.path.join(BASE, "chunks")
IMG_DIR = os.path.join(BASE, "images", "extracted")
CAPTION_MODEL = "Salesforce/blip-image-captioning-base"

captioner = pipeline("image-to-text", model=CAPTION_MODEL)


def sanitize_filename(text: str, max_len: int = 50) -> str:
    slug = re.sub(r"[^A-Za-z0-9 _-]+", "", text).strip().replace(" ", "_")
    return slug[:max_len]


def extract_pdf(pdf_path: str):
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    os.makedirs(TEXT_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

    # 1️⃣ Extract full text from each page
    doc = fitz.open(pdf_path)
    full_text = "".join(page.get_text("text") for page in doc)
    doc.close()

    # 2️⃣ Chunk & dedupe text
    raw_chunks = [
        full_text[i: i + CHUNK_SIZE]
        for i in range(0, len(full_text), CHUNK_SIZE)
    ]
    seen, chunks = set(), []
    for c in raw_chunks:
        chunk = c.strip()
        if chunk and chunk not in seen:
            seen.add(chunk)
            chunks.append(chunk)

    # 3️⃣ Write text chunks
    for idx, chunk in enumerate(chunks, start=1):
        out_path = os.path.join(TEXT_DIR, f"{base}_chunk{idx}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(chunk)

    # 4️⃣ Extract & caption images (skip duplicates)
    doc = fitz.open(pdf_path)
    images = []
    seen_xrefs = set()

    for page_num in range(len(doc)):
        page = doc[page_num]
        for img_index, img_meta in enumerate(page.get_images(full=True), start=1):
            xref = img_meta[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                # Generate filename first
                caption = "extracted_image"  # Default caption
                try:
                    # Try to get caption immediately after extraction
                    raw_name = f"{base}_p{page_num+1}_img{img_index}.png"
                    raw_path = os.path.join(IMG_DIR, raw_name)
                    pix.save(raw_path)

                    # Generate caption
                    caption = captioner(Image.open(raw_path))[
                        0]["generated_text"]
                    safe_slug = sanitize_filename(caption)
                    final_name = f"{base}_p{page_num+1}_img{img_index}_{safe_slug}.png"
                    final_path = os.path.join(IMG_DIR, final_name)
                    os.rename(raw_path, final_path)
                except Exception as e:
                    print(f"⚠️ Could not caption image: {e}")
                    final_name = f"{base}_p{page_num+1}_img{img_index}.png"
                    final_path = os.path.join(IMG_DIR, final_name)
                    pix.save(final_path)

                pix = None

                # ✅ FIX: Use web-friendly relative path from project root
                rel_path = f"images/extracted/{os.path.basename(final_path)}"

                images.append({
                    "path": rel_path,  # This is the key fix!
                    "caption": caption,
                    "page": page_num + 1,
                    "filename": os.path.basename(final_path)
                })
            except Exception as e:
                print(
                    f"⚠️ Could not extract image {xref} from page {page_num+1}: {e}")
                continue

    doc.close()

    # 5️⃣ Save image metadata
    if images:
        meta_file = os.path.join(TEXT_DIR, f"{base}_images.json")
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(images, f, indent=2)

    print(f"✅ {base}: {len(chunks)} text chunks, {len(images)} images extracted")
    return images


if __name__ == "__main__":
    PDF_DIR = os.path.join(BASE, "data", "input_pdfs")

    if not os.path.isdir(PDF_DIR):
        raise FileNotFoundError(
            f"Cannot find PDF folder at {PDF_DIR!r}. Create it and drop your .pdfs there.")

    for fn in os.listdir(PDF_DIR):
        if fn.lower().endswith(".pdf"):
            path = os.path.join(PDF_DIR, fn)
            print(f"Processing {fn}...")
            extract_pdf(path)
