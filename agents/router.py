from agents.guardrails import GuardrailsValidator
from agents.knowledge_base import KnowledgeBase
from agents.web_search import WebSearchAgent
from utils.solution_generator import SolutionGenerator
from config.settings import Settings
import json
import os
from datetime import datetime
from typing import Dict

class MathRoutingAgent:
    def __init__(self):
        self.settings = Settings()
        self.guardrails = GuardrailsValidator()
        self.knowledge_base = KnowledgeBase()
        self.web_search = WebSearchAgent(
            perplexity_api_key=self.settings.PERPLEXITY_API_KEY,
            tavily_api_key=self.settings.TAVILY_API_KEY
        )
        self.solution_generator = SolutionGenerator(hf_token=self.settings.HF_API_TOKEN)
        
        # Initialize knowledge base with error handling
        try:
            self.knowledge_base.initialize_with_dataset()
        except Exception as e:
            print(f"Warning: Knowledge base initialization failed: {e}")
        
        # Feedback storage - ensure directory exists
        os.makedirs("data", exist_ok=True)
        self.feedback_file = "data/feedback.json"
    
    def process_query(self, user_query: str) -> Dict:
        """Main routing logic with better error handling"""
        
        try:
            # Step 1: Input Guardrails
            is_valid, validation_message = self.guardrails.validate_input(user_query)
            if not is_valid:
                return {
                    "success": False,
                    "error": validation_message,
                    "source": "guardrails"
                }
            
            # Step 2: Search Knowledge Base
            try:
                kb_result = self.knowledge_base.search_knowledge_base(user_query)
            except Exception as e:
                print(f"Knowledge base search failed: {e}")
                kb_result = {"found": False}
            
            if kb_result.get("found", False):
                # Found in knowledge base
                solution = self._format_kb_solution(kb_result["solution"])
                source = "knowledge_base"
                
            else:
                # Step 3: Web Search
                try:
                    search_results = self.web_search.search_math_solution(user_query)
                    
                    if search_results["success"]:
                        # Generate solution from web results
                        web_content = self.web_search.extract_solution_content(
                            search_results["results"], 
                            search_results["source"]
                        )
                        solution = self.solution_generator.generate_step_by_step_solution(
                            user_query, web_content
                        )
                        source = f"web_search_{search_results['source']}"
                    else:
                        # Generate solution using AI only
                        solution = self.solution_generator.generate_step_by_step_solution(user_query)
                        source = "ai_generated"
                        
                except Exception as e:
                    print(f"Web search failed: {e}")
                    # Fallback to AI-only generation
                    solution = self.solution_generator.generate_step_by_step_solution(user_query)
                    source = "ai_generated"
            
            # Step 4: Output Guardrails
            try:
                is_valid_output, output_message = self.guardrails.validate_output(solution)
                if not is_valid_output:
                    return {
                        "success": False,
                        "error": output_message,
                        "source": "output_guardrails"
                    }
            except Exception as e:
                print(f"Output validation failed: {e}")
                # Continue without validation if it fails
            
            return {
                "success": True,
                "solution": solution,
                "source": source,
                "confidence": kb_result.get("confidence", 0.5),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Processing failed: {str(e)}",
                "source": "system_error"
            }
    
    def _format_kb_solution(self, kb_solution: Dict) -> str:
        """Format knowledge base solution"""
        if isinstance(kb_solution, dict):
            return f"""**Topic:** {kb_solution.get('topic', 'General')}
**Difficulty:** {kb_solution.get('difficulty', 'Medium')}

**Question:** {kb_solution.get('question', 'N/A')}

**Solution:**
{kb_solution.get('solution', 'No solution available')}

*Source: Knowledge Base*"""
        else:
            return f"**Solution:**\n{str(kb_solution)}\n\n*Source: Knowledge Base*"
    
    def process_feedback(self, query: str, solution: str, feedback: str, rating: int) -> Dict:
        """Handle human feedback for continuous learning"""
        
        try:
            # Refine solution based on feedback
            if rating < 3:  # Poor rating, needs improvement
                refined_solution = self.solution_generator.simplify_solution(solution, feedback)
            else:
                refined_solution = solution
            
            # Store feedback
            feedback_entry = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "original_solution": solution,
                "refined_solution": refined_solution,
                "feedback": feedback,
                "rating": rating
            }
            
            self._save_feedback(feedback_entry)
            
            return {
                "success": True,
                "refined_solution": refined_solution,
                "message": "Thank you for your feedback! The solution has been improved."
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Feedback processing failed: {str(e)}",
                "message": "Sorry, we couldn't process your feedback right now."
            }
    
    def _save_feedback(self, feedback_entry: Dict):
        """Save feedback to file"""
        try:
            try:
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    feedback_data = json.load(f)
            except FileNotFoundError:
                feedback_data = []
            
            feedback_data.append(feedback_entry)
            
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump(feedback_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error saving feedback: {e}")
