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
            print("Dataset not found. Creating sample dataset...")
            # self.create_sample_dataset(dataset_path)
    
    # def create_sample_dataset(self, path: str):
    #     """Create sample math dataset"""
    #     sample_data = [
    #         {
    #             "question": "Solve the quadratic equation x² + 5x + 6 = 0",
    #             "solution": "Step 1: Identify coefficients a=1, b=5, c=6\nStep 2: Use quadratic formula x = (-b ± √(b²-4ac))/2a\nStep 3: x = (-5 ± √(25-24))/2 = (-5 ± 1)/2\nStep 4: Solutions are x = -2 and x = -3"
    #         },
    #         {
    #             "question": "Find the derivative of f(x) = 3x² + 2x + 1",
    #             "solution": "Step 1: Apply power rule to each term\nStep 2: d/dx(3x²) = 6x\nStep 3: d/dx(2x) = 2\nStep 4: d/dx(1) = 0\nStep 5: f'(x) = 6x + 2"
    #         },
    #         {
    #             "question": "Calculate the area of a circle with radius 5",
    #             "solution": "Step 1: Use formula A = πr²\nStep 2: Substitute r = 5\nStep 3: A = π × 5² = 25π\nStep 4: A ≈ 78.54 square units"
    #         }
    #     ]
        
    #     with open(path, 'w') as f:
    #         json.dump(sample_data, f, indent=2)
        
    #     self.vector_store.add_documents(sample_data)
    
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