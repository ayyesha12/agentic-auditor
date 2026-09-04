from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import glob

CORPUS_DIR = "data/corpus"
INDEX_DIR = "data/faiss_index"

# Use a dedicated embedding model — NOT a chat model like qwen3:1.7b
# Chat models don't support embeddings. nomic-embed-text is designed for this.
EMBEDDING_MODEL = "nomic-embed-text"

def build_index():
    print(f"Loading documents from {CORPUS_DIR}...")

    all_docs = []
    txt_files = glob.glob(os.path.join(CORPUS_DIR, "*.txt"))

    if not txt_files:
        print("ERROR: No .txt files found in data/corpus/")
        return

    for filepath in txt_files:
        try:
            loader = TextLoader(filepath, encoding="utf-8")
            docs = loader.load()
            all_docs.extend(docs)
            print(f"  Loaded: {os.path.basename(filepath)}")
        except Exception as e:
            print(f"  WARNING: Could not load {filepath} — {e}")

    print(f"\nLoaded {len(all_docs)} documents total")

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(all_docs)
    print(f"Created {len(chunks)} chunks")

    print(f"Embedding with {EMBEDDING_MODEL}...")
    print("(This takes a few minutes on CPU — do not close the terminal)")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    db = FAISS.from_documents(chunks, embeddings)

    print(f"Saving index to {INDEX_DIR}...")
    os.makedirs(INDEX_DIR, exist_ok=True)
    db.save_local(INDEX_DIR)
    print("\nDone. FAISS index saved successfully.")
    print(f"Index location: {INDEX_DIR}/")

if __name__ == "__main__":
    build_index()