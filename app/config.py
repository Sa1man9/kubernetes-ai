import os
from dotenv import load_dotenv
from pydantic import BaseSettings

load_dotenv()

class AppSettings(BaseSettings):
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")

    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY")
    QDRANT_URL: str = os.getenv("QDRANT_URL") 
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION")

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
    GROQ_FALLBACK_KEY: str = os.getenv("GROQ_FALLBACK_KEY")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL")

    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL")
    FALLBACK_EMBEDDING_MODEL: str = os.getenv("FALLBACK_EMBEDDING_MODEL")

settings = AppSettings()