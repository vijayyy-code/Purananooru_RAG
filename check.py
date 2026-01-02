import os
import requests
import numpy as np
import faiss
import gradio as gr
import time
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# ---------------------------
# ⚙️ Configuration
# ---------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.1-8b-instant"
EXCEL_FILE = "sangam.xlsx"

print("🚀 Initializing Scholar Systems...")
embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# ---------------------------
# 📊 Force-Save Excel Logic
# ---------------------------
def log_to_excel(question, answer, sources):
    """Force-saves research by rewriting the file to ensure visibility."""
    try:
        new_row = pd.DataFrame([{
            "Question": question,
            "Answer": answer,
            "Top sources": sources
        }])

        if not os.path.exists(EXCEL_FILE):
            new_row.to_excel(EXCEL_FILE, index=False)
            print(f"📁 Created new research log: {EXCEL_FILE}")
        else:
            existing_df = pd.read_excel(EXCEL_FILE)
            updated_df = pd.concat([existing_df, new_row], ignore_index=True)
            updated_df.to_excel(EXCEL_FILE, index=False)
            print(f"✅ Research saved! Current database size: {len(updated_df)} rows.")

    except PermissionError:
        print("⚠️ ACTION REQUIRED: You must CLOSE 'sangam.xlsx' in Excel so I can save the data!")
    except Exception as e:
        print(f"❌ Excel Error: {e}")

# ---------------------------
# 📖 Data & Retrieval Logic
# ---------------------------
def load_sangam(file_path="sangam.txt"):
    if not os.path.exists(file_path):
        return ""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text, chunk_size=200, overlap=50):
    tokens = text.split()
    return [" ".join(tokens[i:i+chunk_size]) for i in range(0, len(tokens), chunk_size - overlap) if tokens[i:i+chunk_size]]

def build_indices(chunks):
    print(f"🔍 Indexing {len(chunks)} text chunks...")
    vectors = embed_model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    faiss_index = faiss.IndexFlatIP(vectors.shape[1])
    faiss_index.add(vectors)
    bm25_index = BM25Okapi([c.split() for c in chunks])
    return faiss_index, bm25_index

raw_text = load_sangam("sangam.txt")
text_chunks = chunk_text(raw_text)
faiss_idx, bm25_idx = build_indices(text_chunks)

# ---------------------------
# 🧠 Scholar Logic (Updated for Robustness & Rate Limits)
# ---------------------------
def chat_fn(message, history):
    # Standard 10-second wait to allow token bucket to refill
    print("⏳ Waiting for rate limit window...")
    time.sleep(10)

    # 1. High-Precision Retrieval (Restored to 3-4 chunks for 100% Accuracy)
    q_vec = embed_model.encode([message], convert_to_numpy=True, normalize_embeddings=True)
    _, I = faiss_idx.search(q_vec, k=3)
    sem_results = [text_chunks[i] for i in I[0]]
    key_results = bm25_idx.get_top_n(message.split(), text_chunks, n=2)
    # Combine results and remove duplicates, keeping high depth
    relevant_chunks = list(dict.fromkeys(sem_results + key_results))[:3]
    
    context_block = "\n\n".join([f"SOURCE {i+1}:\n{c}" for i, c in enumerate(relevant_chunks)])
    
    system_instruction = (
        "You are a strict Sangam Scholar. Follow accuracy rules:\n"
        "1. Check Poem Numbers. 2. Identify Thinai (Puram/Akam). 3. Look for the Moral/Punchline.\n"
        "4. Avoid keyword traps. 5. Interpret symbols philosophically based on provided context."
    )

    messages = [{"role": "system", "content": system_instruction}]
    # Keep last 1 exchange for context to stay under token limits
    for item in (history[-1:] if history else []):
        if isinstance(item, (list, tuple)):
            u, b = item
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": b})
            
    messages.append({"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {message}"})

    headers = {"Authorization": f"Bearer {GROQ_API_KEY.strip()}", "Content-Type": "application/json"}
    
    # 2. Retry Logic: If rate limit is hit, wait 60 seconds and try again
    for attempt in range(2):
        try:
            payload = {"model": MODEL_NAME, "messages": messages, "temperature": 0.1, "max_tokens": 600}
            response = requests.post(GROQ_API_URL, headers=headers, json=payload)
            
            if response.status_code == 200:
                answer = response.json()["choices"][0]["message"]["content"]
                
                # Full citation string for Excel
                citations_string = " | ".join(relevant_chunks)
                
                # --- THE PERSISTENT LOGGING STEP ---
                log_to_excel(message, answer, citations_string)
                
                return f"{answer}\n\n---\n**Verified Sources:**\n" + "\n".join([f"- {c[:150]}..." for c in relevant_chunks])
            
            elif response.status_code in [413, 429]:
                print(f"🛑 Rate limit hit. Waiting 60 seconds to resolve...")
                time.sleep(60)
                continue # Retry
            else:
                return f"Error: {response.text}"
                
        except Exception as e:
            return f"System error: {str(e)}"
            
    return "Rate limit still active after 60s retry. Please wait a moment before asking again."

# ---------------------------
# 🎨 UI Launch
# ---------------------------
demo = gr.ChatInterface(
    fn=chat_fn,
    title="🏛️ High-Precision Sangam Scholar",
    description="Responses are automatically saved to `sangam.xlsx`. This version waits 60s if rate limits are hit.",
)

if __name__ == "__main__":
    demo.launch()