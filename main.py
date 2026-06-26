from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from ingestion import load_code_files,chunk_files,embed_and_store
from chain import get_qa_chain, get_doc_chain
import os




app = FastAPI(title = "GitRot")



@app.get("/")
def root():
    return{"message" : "GitRot is running"}

class IngestRequest(BaseModel):
    folder_path: str

class AskRequest(BaseModel):
    question: str  

class DocsRequest(BaseModel):
    filename: str      

@app.post("/ingest")
def ingest(req: IngestRequest):
    if not os.path.exists(req.folder_path):
        raise HTTPException(status_code=400, detail=f"Folder not found: {req.folder_path}")
    
    files = load_code_files(req.folder_path)
    
    if not files:
        raise HTTPException(status_code=400, detail="No supported code files found in that folder.")
    
    chunks = chunk_files(files)
    embed_and_store(chunks)
    return {
        "status": "success",
        "files_ingested": len(files),
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



@app.post("/generate-docs")
def generate_docs(req: DocsRequest):
    try:
        chain, retriever = get_doc_chain()
        docs =chain.invoke(req.filename)
        return{
            "filename": req.filename,
            "documentation": docs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


