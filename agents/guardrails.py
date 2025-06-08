import re
from typing import Tuple

class GuardrailsValidator:
    def __init__(self):
        self.math_keywords = [
            'solve', 'equation', 'algebra', 'calculus', 'geometry', 
            'trigonometry', 'statistics', 'probability', 'derivative',
            'integral', 'limit', 'function', 'graph', 'matrix',
            'vector', 'scalar', 'logarithm', 'exponent', 'inequality',
            'polynomial', 'quadratic', 'linear', 'mean', 'median',
            'mode', 'variance', 'standard deviation', 'binomial',
            'permutation', 'combination', 'series', 'sequence',
            'differential', 'area', 'volume', 'angle', 'radius',
            'pi', 'theorem', 'proof', 'identity', 'domain', 'range',
            'asymptote', 'factor', 'intercept', 'transformation',
            'complex', 'imaginary', 'real', 'root', 'zero'
        ]

        self.blocked_terms = [
            'hack', 'cheat', 'answer key', 'exam paper', 'test solutions'
        ]
    
    def validate_input(self, query: str) -> Tuple[bool, str]:
        """Input guardrails validation"""
        
        if len(query) > 500:
            return False, "Query too long. Please keep it under 500 characters."
        
        query_lower = query.lower()
        has_math_content = any(keyword in query_lower for keyword in self.math_keywords)
        
        if not has_math_content:
            return False, "Please ask mathematics-related questions only."
        
        has_blocked_terms = any(term in query_lower for term in self.blocked_terms)
        if has_blocked_terms:
            return False, "Cannot assist with exam cheating or unauthorized solutions."
        
        return True, "Valid query"
    
    def validate_output(self, response: str) -> Tuple[bool, str]:
        """Output guardrails validation"""
        
        if len(response) < 10:
            return False, "Response too brief for educational content."
        
        # Removed step-by-step format enforcement here.
        
        return True, "Valid response"
