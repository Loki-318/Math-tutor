import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    def __init__(self):
        self.TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
        self.PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
        self.QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.HF_API_TOKEN = os.getenv("HF_API_TOKEN")

        # Guardrails settings
        self.MAX_QUERY_LENGTH = 500
        self.ALLOWED_TOPICS = ["mathematics", "algebra", "calculus", "geometry", "statistics"]

        # Vector DB settings
        self.COLLECTION_NAME = "math_knowledge_base"
        self.VECTOR_SIZE = 384  # for sentence-transformers/all-MiniLM-L6-v2
