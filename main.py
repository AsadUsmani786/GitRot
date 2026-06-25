from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from ingestion import load_code_files,chunk_files,embed_and_store
from chain import get_qa_chain 

app = FastAPI(title = "DevDocs AI")

@app.get("/")
def root():
    return{"message" : "DevDocs AI is running"}

class IngestRequest(BaseModel):
    folder_path: str

class AskRequest(BaseModel):
    question: str    

@app.post("/ingest")
def ingest(req: IngestRequest):
    files = load_code_files(req.folder_path)
    chunks = chunk_files(files)
    result = embed_and_store(chunks)
    return{
        "files found" : len(files),
        "chunks_stored": len(chunks)
    }

@app.post("/ask")
def ask(req: AskRequest):
    try:
        chain, retriever = get_qa_chain()
        answer = chain.invoke(req.question)
        sources = retriever.invoke(req.question)
        return{
            "answer": answer,
            "sources": list(set([doc.metadata["filename"] for doc in sources]))
        }
    except Exception as e:
        raise HTTPException(status_code=500, details=str(e))


