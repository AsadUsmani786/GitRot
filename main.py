from fastapi import FastAPI,HTTPException
from pydantic import BaseModel     #to define the structure of incoming json data
from ingestion import load_code_files,chunk_files,embed_and_store
from chain import get_qa_chain, get_doc_chain
import os




app = FastAPI(title = "GitRot")        #creating the fast api application (variable app represents the api server)



@app.get("/")          #decoder()creating the root endpoint
def root():
    return{"message" : "GitRot is running"}

class IngestRequest(BaseModel):             #what ingest expects (folder path)
    folder_path: str

class AskRequest(BaseModel):              #input for the user (how does the auth work?)
    question: str  

class DocsRequest(BaseModel):             #input for generate-docs
    filename: str      

@app.post("/ingest")
def ingest(req: IngestRequest):          #user sends the folder path
    if not os.path.isdir(req.folder_path):
        raise HTTPException(status_code=400, detail=f"Folder not found: {req.folder_path}")
    
    files = load_code_files(req.folder_path)    #calling the ingestion function 
    
    if not files:
        raise HTTPException(status_code=400, detail="No supported code files found in that folder.")       #if folder exists but no supported files exist
    
    chunks = chunk_files(files)        #chunk the files
    embed_and_store(chunks)           #embed and store
    return {                          #return the ingestion result
        "status": "success",
        "files_ingested": len(files),
        "chunks_stored": len(chunks)
    }

@app.post("/ask")                         #ask endpoint
def ask(req: AskRequest):
    try:
        chain, retriever = get_qa_chain()            #get the qa chain(from chain.py)
        answer = chain.invoke(req.question)
        sources = retriever.invoke(req.question)
        return {
            "answer": answer,
            "sources": list({
                doc.metadata.get("filename") or doc.metadata.get("source") or "unknown"
                for doc in sources
            }),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/generate-docs")                        #generate-docs Endpoint
def generate_docs(req: DocsRequest):
    try:
        chain = get_doc_chain(req.filename)
        docs = chain.invoke(req.filename)
        return{
            "filename": req.filename,
            "documentation": docs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


