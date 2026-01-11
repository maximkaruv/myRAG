import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any
from rag_engine import rag_engine

app = FastAPI(title="Simple RAG System with Metadata")

# Models for API
class DocumentCreate(BaseModel):
    title: str
    content: str
    role: str
    metadata: Any

class DocumentResponse(BaseModel):
    title: str
    content: str
    role: str
    metadata: Any

class Query(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str
    context: List[DocumentResponse]

# Endpoints
@app.post("/documents", response_model=dict)
async def add_document(doc: DocumentCreate):
    try:
        doc_id = rag_engine.add_document(
            title=doc.title,
            content=doc.content,
            role=doc.role,
            metadata=doc.metadata
        )
        return {"status": "success", "document_id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", response_model=List[DocumentResponse])
async def list_documents():
    return rag_engine.documents

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: int):
    success = rag_engine.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "success", "message": f"Document {doc_id} deleted"}

@app.post("/ask", response_model=Answer)
async def ask_question(query: Query):
    try:
        context_docs = rag_engine.search(query.question)
        answer = rag_engine.generate_answer(query.question)
        return Answer(answer=answer, context=context_docs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
