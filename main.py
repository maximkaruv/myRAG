import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from rag_engine import rag_engine

app = FastAPI(title="Simple RAG System")

# Models for API
class Document(BaseModel):
    text: str

class Query(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str
    context: List[str]

# Endpoints
@app.post("/documents", response_model=dict)
async def add_document(doc: Document):
    try:
        doc_id = rag_engine.add_document(doc.text)
        return {"status": "success", "document_id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", response_model=List[str])
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
        context = rag_engine.search(query.question)
        answer = rag_engine.generate_answer(query.question)
        return Answer(answer=answer, context=context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
