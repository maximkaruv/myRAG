import faiss
import numpy as np
from openai import OpenAI
from config import settings

class RAGEngine:
    def __init__(self):
        # Clients
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL)
        self.deepseek_client = OpenAI(api_key=settings.DEEPSEEK_API_KEY, base_url=settings.DEEPSEEK_BASE_URL)
        
        # Vector DB
        self.dimension = settings.VECTOR_DIMENSION
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []  # Store original text documents

    def get_embedding(self, text: str):
        response = self.openai_client.embeddings.create(
            input=text,
            model=settings.EMBEDDING_MODEL
        )
        
        cost = response.usage.total_tokens / 1_000_000 * 0.02 * 80
        import json
        print("OpenAI embeddings cost: ", cost, response.usage.total_tokens, json.dumps(response.model_dump(), indent=2))
        
        return np.array(response.data[0].embedding).astype('float32')

    def add_document(self, text: str):
        embedding = self.get_embedding(text)
        self.index.add(np.array([embedding]))
        self.documents.append(text)
        return len(self.documents) - 1

    def delete_document(self, doc_id: int):
        if 0 <= doc_id < len(self.documents):
            self.documents.pop(doc_id)
            self.rebuild_index()
            return True
        return False

    def rebuild_index(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        if self.documents:
            embeddings = [self.get_embedding(doc) for doc in self.documents]
            self.index.add(np.array(embeddings))

    def search(self, query: str, k: int = 3):
        if not self.documents:
            return []
        query_embedding = self.get_embedding(query)
        distances, indices = self.index.search(np.array([query_embedding]), k)
        
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.documents):
                results.append(self.documents[idx])
        return results

    def generate_answer(self, query: str):
        context_docs = self.search(query)
        context = "\n".join(context_docs)
        
        prompt = f"Context:\n{context}\n\nQuestion: {query}"
        
        response = self.deepseek_client.chat.completions.create(
            model=settings.GENERATIVE_MODEL,
            messages=[
                {"role": "system", "content": settings.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        
        cost_input = response.usage.prompt_tokens / 1_000_000 * 0.028 * 80
        cost_output = response.usage.total_tokens / 1_000_000 * 0.42 * 80
        print("Deepseek cost input: ", cost_input, response.usage.prompt_tokens)
        print("Deepseek cost output: ", cost_output, response.usage.total_tokens)
        
        return response.choices[0].message.content

rag_engine = RAGEngine()
