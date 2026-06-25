import os   # for python to interact with our os
from pathlib import Path 

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
                    

  





#chunking

from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_files(files: list):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 150,
        separators = ["\nclass","\ndef","\n\n","\n",""],
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




#embedding the chunks and storing the vector into chromadb

from langchain_huggingface import HuggingFaceEmbeddings      #why hf embs ?
from langchain_chroma import Chroma

CHROMA_DIR ="./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

def embed_and_store(chunks: list):
    print(f"Embeddings {len(chunks)} chunks ... (first run downloads the model)")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    texts =[chunk["text"] for chunk in chunks]
    metadatas = [{"source": chunk["source"], "filename": chunk["filename"]} for chunk in chunks]

    vectorstore = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    print(f"Stored {len(texts)} chunks in ChromaDB at {CHROMA_DIR}")
    return vectorstore



if __name__ =="__main__":
    files =load_code_files(".")
    chunks = chunk_files(files)
    print(f"Files: {len(files)}, Chunks: {len(chunks)}")
    embed_and_store(chunks)

          







if __name__ == "__main__":
    files = load_code_files(".")
    chunks = chunk_files(files)
    print(f"Files: {len(files)}")
    print(f"Chunks: {len(chunks)}")
    print(f"\nFirst chunk preview:\n{chunks[0]['text'][:300]}")   
                                    