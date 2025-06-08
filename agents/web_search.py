import urllib.parse
from typing import Dict, List, Optional
from duckduckgo_search import DDGS
import requests
import json

class PerplexitySearch:
    """Perplexity AI search implementation"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.perplexity.ai"

    def search(self, query: str, max_results: int = 3) -> Dict:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "llama-3.1-sonar-small-128k-online",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that provides detailed mathematical solutions with step-by-step explanations. Always cite sources when providing information from the web."
                    },
                    {
                        "role": "user",
                        "content": f"Find comprehensive information about: {query}. Provide step-by-step mathematical solution if applicable. Include relevant formulas and examples."
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.2,
                "top_p": 0.9,
                "search_domain_filter": [
                    "wolframalpha.com", "khanacademy.org", "mathworld.wolfram.com", "brilliant.org"
                ],
                "return_citations": True,
                "return_images": False
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                
                # Debug: Print the response structure
                print(f"[DEBUG] Response data keys: {data.keys()}")
                
                if "choices" not in data or not data["choices"]:
                    return {
                        "success": False,
                        "error": "No choices in response",
                        "results": []
                    }

                content = data["choices"][0]["message"]["content"]
                
                # Handle citations more safely
                citations = data.get("citations", [])
                results = []
                
                # Check if citations is actually a list and not empty
                if isinstance(citations, list) and citations:
                    for i, citation in enumerate(citations[:max_results]):
                        # Ensure citation is a dictionary
                        if isinstance(citation, dict):
                            results.append({
                                "title": citation.get("title", f"Mathematical Resource {i+1}"),
                                "url": citation.get("url", ""),
                                "content": content[:500] + "..." if len(content) > 500 else content
                            })
                        else:
                            # If citation is not a dict, create a generic entry
                            results.append({
                                "title": f"Mathematical Resource {i+1}",
                                "url": str(citation) if citation else "",
                                "content": content[:500] + "..." if len(content) > 500 else content
                            })
                else:
                    # No valid citations, create a single result with the content
                    results.append({
                        "title": "Perplexity Mathematical Solution",
                        "url": "https://perplexity.ai",
                        "content": content
                    })

                return {
                    "success": True,
                    "results": results,
                    "source": "perplexity",
                    "full_response": content
                }

            else:
                return {
                    "success": False,
                    "error": f"Perplexity API error: {response.status_code} - {response.text}",
                    "results": []
                }

        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Network error: {str(e)}",
                "results": []
            }
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"JSON decode error: {str(e)}",
                "results": []
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Perplexity search failed: {str(e)}",
                "results": []
            }


class WebSearchAgent:
    def __init__(self, perplexity_api_key: Optional[str] = None, tavily_api_key: Optional[str] = None):
        self.perplexity_api_key = perplexity_api_key
        self.tavily_api_key = tavily_api_key
        self.perplexity = PerplexitySearch(perplexity_api_key) if perplexity_api_key else None

    def search_math_solution(self, query: str) -> Dict:
        """Search for math solutions using Perplexity > Tavily > DuckDuckGo"""

        # Try Perplexity
        if self.perplexity:
            print("[INFO] Trying Perplexity search...")
            perplexity_result = self.perplexity.search(query)
            if perplexity_result["success"]:
                print("[INFO] Perplexity search successful")
                return perplexity_result
            else:
                print(f"[Perplexity Error] {perplexity_result['error']}")

        # Try Tavily
        if self.tavily_api_key:
            print("[INFO] Trying Tavily search...")
            tavily_result = self._tavily_search(query)
            if tavily_result["success"]:
                print("[INFO] Tavily search successful")
                return tavily_result
            else:
                print(f"[Tavily Error] {tavily_result['error']}")

        # Fallback to DuckDuckGo
        print("[INFO] Falling back to DuckDuckGo search...")
        return self._duckduckgo_search(query)

    def _tavily_search(self, query: str) -> Dict:
        """Use Tavily API for search"""
        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=self.tavily_api_key)
            response = client.search(
                query=f"mathematics {query} step by step solution",
                search_depth="advanced",
                max_results=3
            )

            results = response.get("results", [])
            return {
                "success": True,
                "results": results,
                "source": "tavily"
            }

        except ImportError:
            return {
                "success": False,
                "error": "Tavily package not installed. Install with: pip install tavily-python",
                "results": []
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Tavily search failed: {str(e)}",
                "results": []
            }

    def _duckduckgo_search(self, query: str) -> Dict:
        """Fallback search using duckduckgo_search package"""
        try:
            results = []
            with DDGS() as ddgs:
                search_query = f"mathematics {query} step by step solution"
                search_results = ddgs.text(search_query, region='wt-wt', safesearch='off', max_results=3)
                
                for r in search_results:
                    results.append({
                        "title": r.get("title", "Unknown Title"),
                        "url": r.get("href", ""),
                        "content": r.get("body", "No content available")
                    })
                    
            return {
                "success": True,
                "results": results,
                "source": "duckduckgo"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"DuckDuckGo search failed: {str(e)}",
                "results": []
            }

    def extract_solution_content(self, search_results: List[Dict], source: str = None) -> str:
        """Extract and format solution from search results"""

        if not search_results:
            return "No relevant solutions found online."

        if source == "perplexity":
            # For Perplexity, use the full_response if available
            content = search_results[0].get('content', '')
            return f"**Perplexity AI Response:**\n\n{content}\n"

        solution_content = "Based on online resources, here are some hints and reasoning steps:\n\n"
        for i, result in enumerate(search_results[:2], 1):
            title = result.get('title', 'Unknown')
            content = result.get('content', 'No content available')
            url = result.get('url', '')
            
            solution_content += f"**Source {i}: {title}**\n"
            if url:
                solution_content += f"URL: {url}\n"
            solution_content += f"Content: {content}\n\n"

        return solution_content


# Example usage
# if __name__ == "__main__":
#     # Initialize the search agent
#     # Replace with your actual API keys
#     agent = WebSearchAgent(
#         perplexity_api_key="your_perplexity_key_here",
#         tavily_api_key="your_tavily_key_here"
#     )
    
#     # Test search
#     query = "quadratic formula derivation"
#     result = agent.search_math_solution(query)
    
#     if result["success"]:
#         print("Search Results:")
#         print(agent.extract_solution_content(result["results"], result["source"]))
#     else:
#         print(f"Search failed: {result['error']}")