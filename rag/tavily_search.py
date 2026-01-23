import logging
from config import tavily 
from core.schemas import State
from langchain_core.documents import Document
logger = logging.getLogger(__name__)

async def tavily_search(state: State):
    """
    Search the web for additional context using Tavily.
    """
    query = state.get("unanswered") 
    
    if not query:
        return {"retrievedDocs": []}

    logger.info(f"--- WEB SEARCHING: {query} ---")
    
    try:
        response = await tavily.search(
            query=query, 
            search_depth="advanced", 
            max_results=3
        )
        
        search_docs = [
            Document(
                page_content=result["content"],
                metadata={
                    "source": result["url"],
                    "title": result.get("title", "Web Result"),
                    "paper_id": "web_search" 
                }
            )
            for result in response.get("results", [])
        ]
        
        # Merge with existing ArXiv docs if they exist
        existing_docs = state.get("retrievedDocs", [])
        return {"retrievedDocs": existing_docs + search_docs}

    except Exception as e:
        print(f"Tavily search failed: {e}")
        # Fallback: return existing docs without adding web results
        return {"retrievedDocs": state.get("retrievedDocs", [])}