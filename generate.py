# src/generate.py (FIXED VERSION)

import os
import re
import time
from dotenv import load_dotenv
import google.generativeai as genai
from query import retrieve

# ─── Load API Key ─────────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not found in .env")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# ─── Utility ──────────────────────────────────────────────────────────────────


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:50]


def wrap_html(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; max-width: 800px; margin-left: auto; margin-right: auto; }}
    h1, h2 {{ color: #2c3e50; }}
    h1 {{ border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }}
    .image-container {{ text-align: center; margin: 30px 0; }}
    .blog-image {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; padding: 5px; }}
    .image-caption {{ font-style: italic; color: #666; text-align: center; margin-top: 10px; }}
    p {{ line-height: 1.8; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  {body}
</body>
</html>"""


def fix_image_paths(html_content: str, images_dir: str) -> str:
    """Fix image paths in HTML content"""
    # Remove absolute paths
    html_content = re.sub(
        r'[A-Z]:\\[^"]+images\\extracted\\', 'images/extracted/', html_content)
    html_content = html_content.replace(
        "images\\extracted\\", "images/extracted/")

    # Ensure all image paths are relative and web-friendly
    html_content = html_content.replace(
        "D:/Automated Blog Post 2nd version - pymupdf/", "")
    html_content = html_content.replace(
        "D:\\Automated Blog Post 2nd version - pymupdf\\", "")

    return html_content

# ─── Blog Generation ──────────────────────────────────────────────────────────


def generate_blog(topic: str):
    results = retrieve(topic)
    texts = results.get("texts", [])
    images = results.get("images", [])

    print(f"📊 Retrieved {len(texts)} text chunks and {len(images)} images")

    # Debug: Print image paths
    for i, img in enumerate(images):
        print(f"🖼️ Image {i+1}: {img['path']}")

    # Build enhanced prompt for Gemini
    prompt = f"""You are a technical blog writer. Write a detailed HTML blog post titled "{topic}" using the excerpts below.

IMPORTANT INSTRUCTIONS:
1. Use proper HTML tags: <h1>, <h2>, <h3>, <p>, <ul>, <li>, <strong>, <em>
2. Structure the blog with introduction, main content, and conclusion
3. Make the content flow logically
4. If you mention images, use the exact image paths provided below
5. Format images like this: <div class="image-container"><img src="IMAGE_PATH" class="blog-image"><p class="image-caption">CAPTION_TEXT</p></div>

## Retrieved Text Excerpts:
"""
    for i, t in enumerate(texts, 1):
        prompt += f"Excerpt {i} (relevance: {t['score']:.3f}):\n{t['text'].strip()}\n\n"

    if images:
        prompt += "## Available Images (use these where relevant):\n"
        for i, im in enumerate(images, 1):
            prompt += f"Image {i}: {im['caption']} - Path: {im['path']}\n"

    prompt += f"""

Now write the blog post in HTML format. Use the image paths exactly as provided above.

Start with: <h1>{topic}</h1>"""

    # Generate blog content
    response = model.generate_content(prompt)
    return response.text.strip(), images

# ─── Save Output ──────────────────────────────────────────────────────────────


def main():
    topic = input("▶️ Enter your blog topic or question: ").strip()
    if not topic:
        print("No topic provided—exiting.")
        return

    raw_html, images = generate_blog(topic)

    # Fix image paths
    html_cleaned = fix_image_paths(raw_html, "images/extracted")

    # Ensure the HTML is properly structured
    if "<html" not in html_cleaned.lower():
        html_cleaned = wrap_html(topic, html_cleaned)

    # Save to outputs folder
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    fname = f"{slugify(topic)}_{int(time.time())}.html"
    out_path = os.path.join(out_dir, fname)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_cleaned)

    print(f"\n✅ Blog saved to: {out_path}")
    print(f"📊 Used {len(images)} images")
    print("🌐 Open the HTML file in your browser to view the blog")

    # Verify image paths exist
    images_dir = os.path.join(base_dir, "images", "extracted")
    if os.path.exists(images_dir):
        print(f"📁 Images directory: {images_dir}")
    else:
        print("❌ Images directory not found!")


if __name__ == "__main__":
    main()
