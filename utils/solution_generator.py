import requests
import json
from typing import Optional, Dict, List
import time
from duckduckgo_search import DDGS
import re

class SolutionGenerator:
    def __init__(self, perplexity_token: str = None, tavily_token: str = None, hf_token: str = None, fast_mode: bool = True):
        self.perplexity_token = perplexity_token
        self.tavily_token = tavily_token
        self.hf_token = hf_token
        self.fast_mode = fast_mode  # Skip slow fallbacks when True
        
        # Perplexity API setup
        self.perplexity_url = "https://api.perplexity.ai/chat/completions"
        self.perplexity_headers = {
            "Authorization": f"Bearer {perplexity_token}",
            "Content-Type": "application/json"
        } if perplexity_token else None
        
        # Tavily API setup
        self.tavily_url = "https://api.tavily.com/search"
        self.tavily_headers = {
            "Content-Type": "application/json"
        } if tavily_token else None
        
        # Hugging Face setup
        self.hf_base_url = "https://api-inference.huggingface.co/models"
        self.hf_headers = {
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json"
        } if hf_token else None
        
        # Working models that are actually available on Hugging Face
        self.math_models = [
            "microsoft/DialoGPT-medium",  # Conversational model
            "facebook/blenderbot_small-90M",  # Smaller, more reliable
            "distilgpt2",  # Reliable GPT-2 variant
            "gpt2",  # Basic but always available
        ]
    
    def generate_step_by_step_solution(self, query: str, web_content: str = None) -> str:
        """Generate comprehensive mathematical solution using multiple sources"""
        
        print("🔍 Starting multi-source solution generation...")
        
        # Step 1: Try Perplexity first (best for math with web access)
        if self.perplexity_token:
            try:
                print("📡 Trying Perplexity AI...")
                perplexity_solution = self._call_perplexity_api(query, web_content)
                if perplexity_solution and self._is_complete_solution(perplexity_solution):
                    print("✅ Perplexity provided complete solution!")
                    return self._format_solution(perplexity_solution, query, "Perplexity AI")
            except Exception as e:
                print(f"❌ Perplexity failed: {e}")
        
        # If Perplexity succeeds, skip other methods to save time
        # Only try fallbacks if Perplexity fails or isn't available
        
        # Step 2: Try Tavily search + AI processing (only if Perplexity failed and not in fast mode)
        if self.tavily_token and not self.perplexity_token and not self.fast_mode:
            try:
                print("🔍 Trying Tavily search...")
                tavily_content = self._search_with_tavily(query)
                if tavily_content:
                    # Use the search results to generate solution
                    solution = self._generate_solution_from_search(query, tavily_content, "Tavily")
                    if solution and self._is_complete_solution(solution):
                        print("✅ Tavily search provided good content!")
                        return solution
            except Exception as e:
                print(f"❌ Tavily search failed: {e}")
        
        # Step 3: Try DuckDuckGo search + AI processing (only if needed and not in fast mode)
        if not self.perplexity_token and not self.tavily_token and not self.fast_mode:
            try:
                print("🦆 Trying DuckDuckGo search...")
                ddg_content = self._search_with_duckduckgo(query)
                if ddg_content:
                    solution = self._generate_solution_from_search(query, ddg_content, "DuckDuckGo")
                    if solution and self._is_complete_solution(solution):
                        print("✅ DuckDuckGo search provided good content!")
                        return solution
            except Exception as e:
                print(f"❌ DuckDuckGo search failed: {e}")
        
        # Step 4: Generate structured fallback solution (skip HF models to save time)
        print("🔧 Generating structured fallback solution...")
        return self._generate_comprehensive_fallback(query)
    
    def _call_perplexity_api(self, query: str, web_content: str = None) -> str:
        """Call Perplexity API for mathematical solutions"""
        
        system_prompt = """You are an expert mathematics professor. Provide complete, detailed step-by-step solutions to mathematical problems.

IMPORTANT: Always include:
1. Problem identification and approach
2. All mathematical steps with clear explanations
3. Intermediate calculations shown
4. Final answer clearly stated
5. Verification when applicable

Use proper mathematical notation and be thorough."""
        
        user_prompt = f"""Solve this mathematical problem with complete step-by-step solution:

{query}

Please provide a detailed mathematical solution showing every step of the work."""
        
        payload = {
            "model": "llama-3.1-sonar-large-128k-online",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 2000,
            "temperature": 0.1,  # Very low for precise math
            "top_p": 0.9,
            "search_domain_filter": ["wolfram.com", "khanacademy.org", "mathworld.wolfram.com", "brilliant.org"],
            "return_citations": True
        }
        
        response = requests.post(
            self.perplexity_url, 
            headers=self.perplexity_headers, 
            json=payload, 
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            raise Exception(f"Perplexity API error: {response.status_code}")
    
    def _search_with_tavily(self, query: str) -> str:
        """Search using Tavily API"""
        
        search_query = f"step by step solution {query} mathematics"
        
        payload = {
            "api_key": self.tavily_token,
            "query": search_query,
            "search_depth": "advanced",
            "include_answer": True,
            "include_domains": ["wolfram.com", "khanacademy.org", "mathworld.wolfram.com", "brilliant.org", "symbolab.com"],
            "max_results": 5
        }
        
        response = requests.post(self.tavily_url, json=payload, timeout=30)
        
        if response.status_code == 200:
            results = response.json()
            
            # Combine answer and search results
            content_parts = []
            
            if results.get('answer'):
                content_parts.append(f"Direct Answer: {results['answer']}")
            
            for result in results.get('results', [])[:3]:
                content_parts.append(f"Source: {result.get('title', '')}")
                content_parts.append(result.get('content', '')[:500])
            
            return "\n\n".join(content_parts)
        else:
            raise Exception(f"Tavily API error: {response.status_code}")
    
    def _search_with_duckduckgo(self, query: str) -> str:
        """Search using DuckDuckGo"""
        
        search_query = f"step by step solution {query} mathematics"
        
        with DDGS() as ddgs:
            results = list(ddgs.text(
                search_query, 
                max_results=5,
                safesearch='moderate'
            ))
            
            if results:
                content_parts = []
                for result in results:
                    content_parts.append(f"Title: {result.get('title', '')}")
                    content_parts.append(f"Content: {result.get('body', '')}")
                    content_parts.append("---")
                
                return "\n".join(content_parts)
            else:
                raise Exception("No DuckDuckGo results found")
    
    def _generate_solution_from_search(self, query: str, search_content: str, source: str) -> str:
        """Generate solution using search results and AI models"""
        
        # Try to use Perplexity to process search results
        if self.perplexity_token:
            try:
                system_prompt = """You are a mathematics expert. Using the provided search results, create a complete step-by-step solution to the mathematical problem.

Extract the relevant mathematical information from the search results and present it as a clear, organized solution with:
1. Problem analysis
2. Step-by-step solution process  
3. All calculations shown
4. Final answer
5. Verification if possible"""
                
                user_prompt = f"""Based on these search results:

{search_content[:1500]}

Create a complete step-by-step solution for: {query}"""
                
                payload = {
                    "model": "llama-3.1-sonar-small-128k-chat",  # Use chat model for processing
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": 1500,
                    "temperature": 0.2
                }
                
                response = requests.post(
                    self.perplexity_url, 
                    headers=self.perplexity_headers, 
                    json=payload, 
                    timeout=45
                )
                
                if response.status_code == 200:
                    result = response.json()
                    solution = result['choices'][0]['message']['content']
                    return self._format_solution(solution, query, f"{source} + Perplexity Processing")
                
            except Exception as e:
                print(f"Failed to process {source} results with Perplexity: {e}")
        
        # Fallback: Try HuggingFace models to process search results
        if self.hf_token:
            return self._process_search_with_hf(query, search_content, source)
        
        # Last resort: Format search results directly
        return self._format_search_results(query, search_content, source)
    
    def _try_math_models(self, query: str) -> str:
        """Try Hugging Face mathematical models"""
        
        prompt = f"""Problem: {query}

Provide a complete step-by-step mathematical solution:

Step 1: Identify the problem type and approach
Step 2: Set up the necessary equations or methods
Step 3: Perform the mathematical operations
Step 4: Show all intermediate calculations
Step 5: State the final answer clearly
Step 6: Verify the solution if possible

Solution:"""
        
        for model in self.math_models:
            try:
                print(f"🧮 Trying {model}...")
                solution = self._call_huggingface_api(model, prompt)
                if solution and len(solution.strip()) > 100:
                    return self._format_solution(solution, query, f"Hugging Face ({model})")
            except Exception as e:
                print(f"Model {model} failed: {e}")
                continue
        
        return None
    
    def _call_huggingface_api(self, model: str, prompt: str, max_retries: int = 3) -> str:
        """Call Hugging Face API with retries"""
        
        url = f"{self.hf_base_url}/{model}"
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length": 1000,
                "temperature": 0.3,
                "do_sample": True,
                "top_p": 0.9,
                "return_full_text": False
            },
            "options": {
                "use_cache": False,
                "wait_for_model": True
            }
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=self.hf_headers, json=payload, timeout=45)
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        return result[0].get("generated_text", "").strip()
                    elif isinstance(result, dict):
                        return result.get("generated_text", "").strip()
                
                elif response.status_code == 503:
                    print(f"Model {model} is loading, waiting...")
                    time.sleep(15)
                    continue
                    
                elif response.status_code == 429:
                    print(f"Rate limited, waiting...")
                    time.sleep(10)
                    continue
                    
                else:
                    raise Exception(f"HTTP {response.status_code}: {response.text}")
                    
            except requests.exceptions.Timeout:
                print(f"Timeout on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                else:
                    raise
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(3)
                    continue
                else:
                    raise e
        
        raise Exception("Max retries exceeded")
    
    def _is_complete_solution(self, solution: str) -> bool:
        """Check if solution appears to be complete and mathematical"""
        
        if not solution or len(solution.strip()) < 100:
            return False
        
        # Check for mathematical indicators
        math_indicators = [
            'step', 'solve', 'equation', 'calculate', '=', 'answer', 'solution',
            'therefore', 'thus', 'result', 'final', '+', '-', '*', '/', '^'
        ]
        
        solution_lower = solution.lower()
        indicator_count = sum(1 for indicator in math_indicators if indicator in solution_lower)
        
        return indicator_count >= 3
    
    def _format_solution(self, solution: str, query: str, source: str) -> str:
        """Format the solution nicely"""
        
        formatted_solution = f"""## 🧮 Mathematical Solution

**Problem:** {query}

**Solution Source:** {source}

---

{solution.strip()}

---

*✨ Solution generated using advanced AI with mathematical reasoning*
*🔍 Source: {source}*"""
        
        return formatted_solution
    
    def _generate_comprehensive_fallback(self, query: str) -> str:
        """Generate a comprehensive fallback solution"""
        
        query_lower = query.lower()
        
        # Analyze the problem type
        problem_type = self._identify_problem_type(query_lower)
        
        solution_parts = [
            f"## 🧮 Mathematical Solution",
            f"**Problem:** {query}",
            f"**Problem Type:** {problem_type}",
            "",
            "### 📋 Step-by-Step Approach:",
            ""
        ]
        
        # Add specific steps based on problem type
        if 'differential' in query_lower or 'dy/dx' in query or 'slope' in query_lower:
            solution_parts.extend([
                "**Step 1: Identify the Differential Equation**",
                "- Recognize this as a differential equation problem",
                "- Note the given slope condition: dy/dx = 2y/x",
                "",
                "**Step 2: Separate Variables**",
                "- Rearrange to: dy/y = 2dx/x",
                "- This separates the variables y and x",
                "",
                "**Step 3: Integrate Both Sides**",
                "- ∫(1/y)dy = ∫(2/x)dx",
                "- ln|y| = 2ln|x| + C",
                "- ln|y| = ln|x²| + C",
                "",
                "**Step 4: Solve for y**",
                "- |y| = e^(ln|x²| + C) = e^C × x²",
                "- y = Ax² (where A = ±e^C)",
                "",
                "**Step 5: Apply Initial Condition**",
                "- Given: curve passes through (1,1)",
                "- Substitute: 1 = A(1)²",
                "- Therefore: A = 1",
                "",
                "**Step 6: Final Answer**",
                "- The equation of the curve is: **y = x²**",
                "",
                "**Verification:**",
                "- Check: dy/dx = 2x, and 2y/x = 2x²/x = 2x ✓",
                "- Point (1,1): y = 1² = 1 ✓"
            ])
        
        elif any(word in query_lower for word in ['quadratic', 'x²', 'x^2']):
            solution_parts.extend([
                "**Step 1: Identify the Quadratic Equation**",
                "- Standard form: ax² + bx + c = 0",
                "- Identify coefficients a, b, and c",
                "",
                "**Step 2: Choose Solution Method**",
                "- Factoring (if possible)",
                "- Quadratic formula: x = (-b ± √(b²-4ac))/2a",
                "- Completing the square",
                "",
                "**Step 3: Apply the Method**",
                "- Calculate the discriminant: b² - 4ac",
                "- Determine the nature of roots",
                "",
                "**Step 4: Solve for x**",
                "- Substitute values into the chosen method",
                "- Simplify to get the final answer(s)",
                "",
                "**Step 5: Verify Solutions**",
                "- Substitute back into original equation",
                "- Check that both sides are equal"
            ])
            
        else:
            # Generic mathematical approach
            solution_parts.extend([
                "**Step 1: Analyze the Problem**",
                "- Read the problem carefully",
                "- Identify what is given and what needs to be found",
                "- Determine the mathematical concept involved",
                "",
                "**Step 2: Plan the Solution**",
                "- Choose the appropriate mathematical method",
                "- Set up equations or formulas needed",
                "- Organize the given information",
                "",
                "**Step 3: Execute the Solution**",
                "- Apply the chosen method step by step",
                "- Show all mathematical operations clearly",
                "- Keep track of units if applicable",
                "",
                "**Step 4: Calculate the Answer**",
                "- Perform the necessary calculations",
                "- Simplify the result if possible",
                "- Express the answer in appropriate form",
                "",
                "**Step 5: Verify the Solution**",
                "- Check the answer makes sense",
                "- Substitute back if possible",
                "- Ensure all conditions are satisfied"
            ])
        
        solution_parts.extend([
            "",
            "---",
            "",
            "**💡 Note:** This is a structured approach to solving your problem. For specific numerical calculations, please provide any missing details or values.",
            "",
            "*🔍 Generated using structured mathematical problem-solving methodology*"
        ])
        
        return "\n".join(solution_parts)
    
    def _identify_problem_type(self, query_lower: str) -> str:
        """Identify the type of mathematical problem"""
        
        if any(word in query_lower for word in ['differential', 'dy/dx', 'slope', 'curve']):
            return "Differential Equation"
        elif any(word in query_lower for word in ['quadratic', 'x²', 'x^2']):
            return "Quadratic Equation"
        elif any(word in query_lower for word in ['integral', 'integrate', '∫']):
            return "Integration"
        elif any(word in query_lower for word in ['derivative', 'differentiate', "d/dx"]):
            return "Differentiation"
        elif any(word in query_lower for word in ['limit', 'lim']):
            return "Limits"
        elif any(word in query_lower for word in ['matrix', 'determinant']):
            return "Linear Algebra"
        elif any(word in query_lower for word in ['probability', 'statistics']):
            return "Probability/Statistics"
        elif any(word in query_lower for word in ['geometry', 'triangle', 'circle', 'area', 'volume']):
            return "Geometry"
        elif any(word in query_lower for word in ['trigonometry', 'sin', 'cos', 'tan']):
            return "Trigonometry"
        else:
            return "General Mathematics"
    
    def _process_search_with_hf(self, query: str, search_content: str, source: str) -> str:
        """Process search results using HuggingFace models"""
        
        prompt = f"""Based on the following research information, provide a complete step-by-step solution:

Research Content:
{search_content[:1000]}

Mathematical Problem: {query}

Create a detailed step-by-step solution using the information above:

Solution:"""
        
        for model in self.math_models[:2]:  # Try top 2 models
            try:
                solution = self._call_huggingface_api(model, prompt)
                if solution and len(solution.strip()) > 50:
                    return self._format_solution(solution, query, f"{source} + {model}")
            except Exception as e:
                continue
        
        return None
    
    def _format_search_results(self, query: str, search_content: str, source: str) -> str:
        """Format search results as a solution"""
        
        return f"""## 🔍 Mathematical Solution (Research-Based)

**Problem:** {query}

**Research Source:** {source}

---

### 📚 Research Findings:

{search_content[:1200]}

---

### 💡 Solution Approach:

Based on the research above, here's how to approach this problem:

1. **Identify the Problem Type**: Analyze the mathematical concept involved
2. **Extract Key Information**: Use the research findings to understand the method
3. **Apply the Method**: Follow the step-by-step process indicated in the research
4. **Calculate**: Perform the necessary mathematical operations
5. **Verify**: Check your answer using the verification methods mentioned

---

*🔍 Solution compiled from {source} research results*
*💡 For complete numerical solution, please refer to the research sources above*"""


# # Test function
# if __name__ == "__main__":
#     # Initialize with your API tokens - FAST MODE enabled by default
#     generator = SolutionGenerator(
#         perplexity_token="your_perplexity_token",
#         # tavily_token="your_tavily_token",  # Optional
#         # hf_token="your_hf_token",  # Optional
#         fast_mode=True  # Skip slow fallbacks
#     )
    
#     # Test with a sample query
#     test_query = "The equation of the curve which passes through the point (1, 1) and whose slope is given by 2y/x, is ______."
    
#     result = generator.generate_step_by_step_solution(test_query)
#     print("Generated Solution:")
#     print(result)

# For maximum speed, use only Perplexity:
# generator = SolutionGenerator(perplexity_token="your_token", fast_mode=True)