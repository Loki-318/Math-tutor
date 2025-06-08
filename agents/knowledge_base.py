from utils.vector_store import VectorStore
from typing import List, Dict, Optional
import json

class KnowledgeBase:
    def __init__(self):
        self.vector_store = VectorStore()
        self.similarity_threshold = 0.7
    
    def initialize_with_dataset(self, dataset_path: str = "data/math_dataset.json"):
        """Load and index math dataset"""
        try:
            with open("path_to_dataset.json", encoding="utf-8") as f:
                dataset = json.load(f)
            
            self.vector_store.add_documents(dataset)
            print(f"Loaded {len(dataset)} math problems into knowledge base")
            
        except FileNotFoundError:
            print("Dataset not found.")
    
    def search_knowledge_base(self, query: str) -> Optional[Dict]:
        """Search for relevant solutions in knowledge base"""
        results = self.vector_store.search_similar(query, limit=1)
        
        if results and results[0].score > self.similarity_threshold:
            return {
                "found": True,
                "solution": results[0].payload,
                "confidence": results[0].score
            }
        
        return {"found": False, "solution": None, "confidence": 0}
