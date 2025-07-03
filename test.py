import os
import pickle
import faiss
import openai
from sentence_transformers import SentenceTransformer

# --- File paths ---
openai_key_path = "C:/Users/Steffen/Desktop/Weiterbildung/openai_key.txt"
text_path       = "C:/Users/Steffen/Dropbox/lex-privat.txt"
embedding_path  = "C:/Users/Steffen/code/rag2/cache/embeddings.pkl"
index_path      = "C:/Users/Steffen/code/rag2/cache/index.faiss"

# Ensure the cache directory exists
os.makedirs(os.path.dirname(embedding_path), exist_ok=True)

# --- Load OpenAI API key ---
with open(openai_key_path, "r") as f:
    openai.api_key = f.read().strip()

# --- Load embedding model ---
model = SentenceTransformer("all-MiniLM-L6-v2")     # only loaded once, can be reused

# Ask user whether to rebuild embeddings
rebuild = input("Recalculate embeddings? (yes/no): ").strip().lower() in ["yes", "y"]

# --- Rebuild or load cached embeddings and index ---
if rebuild or not os.path.exists(embedding_path) or not os.path.exists(index_path):
    print("Recalculating embeddings...")

    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = text.split("\n\n")  # simple paragraph-based chunking

    embeddings = model.encode(chunks, show_progress_bar=True)

    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(embeddings)

    with open(embedding_path, "wb") as f:
        pickle.dump(chunks, f)
    faiss.write_index(index, index_path)

    print("Embeddings and index saved.")
else:
    print("Loading cached embeddings and index...")
    with open(embedding_path, "rb") as f:
        chunks = pickle.load(f)
    index = faiss.read_index(index_path)

# --- Ask user a question ---
query = input("\nYour question: ").strip()
if not query:
    print("No question entered. Exiting.")
    exit(1)

# --- Search for relevant chunks ---
query_vec = model.encode([query])
D, I = index.search(query_vec, k=3)

retrieved = "\n\n".join([chunks[i] for i in I[0]])

# --- Send to OpenAI with context ---
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "Answer the question based on the provided context."},
        {"role": "user", "content": f"Context:\n{retrieved}\n\nQuestion: {query}"}
    ],
    temperature=0.2
)

# --- Output answer ---
print("\nGPT Answer\n" + "-"*50)
print(response.choices[0].message["content"].strip())
print("-"*50)
