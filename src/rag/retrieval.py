import logging
from typing import List, Dict
from core.schemas import State, RuntimeContext
from langgraph.runtime import Runtime 
from langchain_core.documents import Document
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

logger = logging.getLogger(__name__)

async def retrieve(state: State, runtime: Runtime[RuntimeContext]) -> Dict[str, List]:
    """
    Retrieve relevant chunks. Uses Grouped Search for diversity if IDs are known,
    otherwise falls back to standard similarity search using native Qdrant calls.
    """
    user_id = runtime.context.user_id
    vectorstore = runtime.context.vectorstore
    settings = runtime.context.settings
    
    score_threshold = settings.get("retrieval_score_threshold", 0.4)
    total_top_k = settings.get("retrieval_top_k", 5)
    
    query = state.get("rewrittenQuestion", "")
    arxiv_ids = state.get("arxivIDs", [])
    
    logger.info(f"Retrieval Query: {query} | Scoping to Papers: {arxiv_ids}")
    
    if not query or len(query) < 2:
        return {"retrievedDocs": [], "confidenceScores": []}

    # 1. Base Filters
    conditions = [FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id))]
    retrieved_docs = []
    confidence_scores = []

    query_vector = await vectorstore.embeddings.aembed_query(query)

    # 2. Branching Logic
    if arxiv_ids:

        # --- Grouped search: ensure paper representativity ---
        conditions.append(FieldCondition(key="metadata.paper_id", match=MatchAny(any=arxiv_ids)))
        group_limit = 3 if len(arxiv_ids) > 1 else total_top_k

        search_result = await vectorstore.client.query_points_groups(
            collection_name=vectorstore.collection_name, 
            query=query_vector,
            group_by="metadata.paper_id",
            limit=len(arxiv_ids),   # nb of distinct groups (papers)
            group_size=group_limit, # nb of chunks per group
            query_filter=Filter(must=conditions),
            score_threshold=score_threshold,
            with_payload=True
        )
        for group in search_result.groups:
            for hit in group.hits:
                # Map native Qdrant point back to LangChain Document
                payload = hit.payload #page content and metadata 
                retrieved_docs.append(Document(
                    page_content=payload.get("page_content", ""),
                    metadata=payload.get("metadata", {})
                ))
                confidence_scores.append(hit.score)
    
    if not retrieved_docs:
        if arxiv_ids:
            logger.info("Grouped search yielded no results. Falling back to global search.")
        else:
            logger.info("No specific papers scoped. Performing standard similarity search.")
        
        # --- Standard similarity search ---
        search_result = await vectorstore.client.query_points(
            collection_name=vectorstore.collection_name,
            query=query_vector,
            query_filter=Filter(must=conditions),
            limit=total_top_k,
            score_threshold=score_threshold,
            with_payload=True
        )
        for hit in search_result.points:
            payload = hit.payload
            retrieved_docs.append(Document(
                page_content=payload.get("page_content", ""),
                metadata=payload.get("metadata", {})
            ))
            confidence_scores.append(hit.score)
    
    logger.info(f"Retrieved {len(retrieved_docs)} docs with confidence scores {confidence_scores}")
    return {
        "retrievedDocs": retrieved_docs,
        "confidenceScores": confidence_scores,
    }