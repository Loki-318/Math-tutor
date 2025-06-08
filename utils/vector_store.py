from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import json
from typing import List, Dict
import uuid

class VectorStore:
    def __init__(self, url: str = "http://localhost:6333", collection_name: str = "math_knowledge_base"):
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.setup_collection()
    
    def setup_collection(self):
        """Initialize Qdrant collection"""
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
        except Exception as e:
            print(f"Collection might already exist: {e}")
    
    def add_documents(self, documents: List[Dict]):
        """Add math problems and solutions to vector store"""
        points = []
        
        for doc in documents:
            # Combine question and solution for embedding
            text = f"Question: {doc['question']} Solution: {doc['solution']}"
            vector = self.encoder.encode(text).tolist()
            
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "question": doc['question'],
                    "solution": doc['solution'],
                    "topic": doc.get('topic', 'general'),
                    "difficulty": doc.get('difficulty', 'medium')
                }
            )
            points.append(point)
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def search_similar(self, query: str, limit: int = 3):
        """Search for similar math problems"""
        query_vector = self.encoder.encode(query).tolist()
        
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True
        )
        
        return search_result