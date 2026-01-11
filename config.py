import os
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: str = Field(default="your_openai_api_key", env="OPENAI_API_KEY")
    DEEPSEEK_API_KEY: str = Field(default="your_deepseek_api_key", env="DEEPSEEK_API_KEY")
    
    # Base URLs
    OPENAI_BASE_URL: str = Field(default="https://api.openai.com/v1", env="OPENAI_BASE_URL")
    DEEPSEEK_BASE_URL: str = Field(default="https://api.deepseek.com", env="DEEPSEEK_BASE_URL")
    
    # Models
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-small", env="EMBEDDING_MODEL")
    GENERATIVE_MODEL: str = Field(default="deepseek-chat", env="GENERATIVE_MODEL")
    
    # RAG Settings
    SYSTEM_PROMPT: str = "You are a helpful assistant. Use the provided context (including metadata like title, source, and role) to answer the user's question. If you don't know the answer, say that you don't know."
    VECTOR_DIMENSION: int = 1536  # Default for text-embedding-3-small
    
    # Storage Settings
    INDEX_PATH: str = "database/faiss_index.bin"
    DOCS_PATH: str = "database/documents.json"
    
    class Config:
        env_file = ".env"

settings = Settings()

