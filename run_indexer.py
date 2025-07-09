import os
from vector_store import load_and_index_documents_by_file


index_path = "faiss_index_by_file"

if not os.path.exists(index_path) or not os.path.exists(os.path.join(index_path, "index.faiss")):
    print("[INFO] FAISS index not found. Creating it now...")
    load_and_index_documents_by_file()
    print("[DONE] Index created successfully.")
else:
    print("[INFO] FAISS index already exists. No action needed.")
