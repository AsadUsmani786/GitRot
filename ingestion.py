import os
import shutil
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import CHROMA_DIR, EMBED_MODEL

SUPPORTED_EXTENSIONS = [".py", ".js", ".ts", ".c", ".java", ".cpp", ".go", ".md"]


def load_code_files(folder_path: str):
    files = []
    for root, _, filenames in os.walk(folder_path):
        for filename in filenames:
            ext = Path(filename).suffix
            if ext in SUPPORTED_EXTENSIONS:
                full_path = os.path.join(root, filename)
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                if content.strip():
                    files.append({
                        "content": content,
                        "source": full_path,
                        "filename": filename,
                    })
                    print(f"Read: {filename} - {len(content)} chars")
    return files


def chunk_files(files: list):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\nclass", "\ndef", "\n\n", "\n", ""],
    )

    chunks = []
    for file in files:
        pieces = splitter.split_text(file["content"])
        for piece in pieces:
            chunks.append({
                "text": piece,
                "source": file["source"],
                "filename": file["filename"],
            })
    return chunks


def embed_and_store(chunks: list):
    print(f"Embeddings {len(chunks)} chunks ... (first run downloads the model)")

    persist_path = Path(CHROMA_DIR)
    if persist_path.exists():
        shutil.rmtree(persist_path)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    texts = [chunk["text"] for chunk in chunks]
    metadatas = [{"source": chunk["source"], "filename": chunk["filename"]} for chunk in chunks]

    vectorstore = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    print(f"Stored {len(texts)} chunks in ChromaDB at {CHROMA_DIR}")
    return vectorstore
