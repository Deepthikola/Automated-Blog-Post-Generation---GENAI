# Automated-Blog-Post-Generation---GENAI
Automated Blog Post Generation - GENAI RAG pipeline

Writing blogs manually is time-intensive and repetitive.​

Creating engaging blogs requires both text and images for better readability.​Readers often prefer a concise summary instead of reading full-length content.​
There is no streamlined way to automatically generate blogs with summaries and relevant images.​

# Workflow
📥 Inputs  ​

   ↓  ​

📝 Preprocessing  ​

   ↓  ​

🖼️ Image Captioning (BLIP)  ​

   ↓  ​

🔢 Embedding (Sentence-BERT)  ​

   ↓  ​

📂 FAISS Indexing (Text + Image)  ​

   ↓  ​

🔍 Retrieval (Relevant Text + Images)  ​

   ↓  ​

🧩 Context Building (Text + Captions + Query)  ​

   ↓  ​

✍️ Blog Generation (HF_Hub: Zephyr-3B/ Propreitary API: gemini-1.5-flash LLM
/ Ollama: Llama2, Gemma 3-1B)  ​

   ↓   ​

📝 Summary Generation (Zephyr-3B)  ​

   ↓  ​

🌐 Postprocessing & Export (HTML with Images)  ​

   ↓  ​

✅ Accuracy / Quality Check  ​
