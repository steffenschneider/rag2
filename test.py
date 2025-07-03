# pip install openai sentence-transformers faiss-cpu tiktoken
import os
import openai
import pickle
import faiss
import tiktoken
from sentence_transformers import SentenceTransformer

# ----------------------- Configuration -----------------------

openai_key_path = "C:/Users/Steffen/Desktop/Weiterbildung/openai_key.txt"
text_path       = "C:/Users/Steffen/Dropbox/lex-privat.txt"
embedding_path  = "C:/Users/Steffen/code/rag2/cache/embeddings.pkl"
index_path      = "C:/Users/Steffen/code/rag2/cache/index.faiss"

chunk_token_limit = 500  # Number of tokens per chunk
chunk_overlap = 50       # Overlap in tokens between chunks
retrieval_k = 10         # Number of similar chunks to retrieve
model_name = "all-MiniLM-L6-v2"

# ----------------------- Helper Functions -----------------------

def load_api_key(path):
    with open(path, "r") as f:
        return f.read().strip()

def tokenize_text(text, max_tokens=chunk_token_limit, overlap=chunk_overlap):
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(text)
    chunks = []

    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk = enc.decode(tokens[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap  # slide forward with overlap
    return chunks

# ----------------------- Load or Build Embeddings -----------------------

os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
openai.api_key = load_api_key(openai_key_path)
model = SentenceTransformer(model_name)

rebuild = input("Recalculate embeddings? (yes/no): ").strip().lower() in ["yes", "y"]

if rebuild or not os.path.exists(embedding_path) or not os.path.exists(index_path):
    print("Reading and chunking text...")
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = tokenize_text(text)
    print(f"Total chunks: {len(chunks)}")

    print("Encoding embeddings...")
    embeddings = model.encode(chunks, show_progress_bar=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
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

# ----------------------- Question Loop -----------------------

while True:
    query = input("\nYour question (or 'exit' to quit): ").strip()
    if query.lower() in ["exit", "quit", "q"]:
        break
    if not query:
        continue

    query_vec = model.encode([query])
    D, I = index.search(query_vec, k=retrieval_k)
    context = "\n\n".join([chunks[i] for i in I[0]])

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert assistant. Use the following context to answer the question as clearly and accurately as possible."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
        ],
        temperature=0.1,
        max_tokens=512
    )

    print("\nAnswer\n" + "-"*50)
    print(response.choices[0].message["content"].strip())
    print("-"*50)
