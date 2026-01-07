import json 
from pathlib import Path
from rapidfuzz import fuzz
from typing import List, Dict

def FuzzyMagic(metadata_json: Dict[str, Dict], query_hints: Dict, top_n: int = 4) -> List[str]:
    """
    Select top-N paper IDs based on fuzzy matching between query hints and paper metadata.
    Args:
        metadata_json: {paper_id: {Title, Authors, Summary, Published, ...}}
        query_hints: {'titles': [...], 'authors': [...], 'topics': [...], 'publicationYears': [...]}
        top_n: number of top papers to return
    Returns:
        List of top-N paper IDs ranked by fuzzy matching score
    """
    # Return empty if no hints provided
    if not any(query_hints.get(k) for k in ["titles", "authors", "topics", "publicationYears"]):
        return []
    
    # Weights for each field
    weights = {
        "titles": 5.0,
        "topics": 3.0,
        "authors": 1.5,
        "publicationYears": 1.0
    }
    q_titles = query_hints.get("titles", [])
    q_topics = query_hints.get("topics", [])
    q_authors = query_hints.get("authors", [])
    q_years = query_hints.get("publicationYears", [])
    
    # Minimum score threshold (0-1 scale after normalization)
    MIN_THRESHOLD = 0.1
    scores = {}
    for paper_id, paper_data in metadata_json.items():
        total_score = 0.0
        max_possible_score = 0.0  # For normalization
        
        # --- Fuzzy match titles ---
        if q_titles:
            paper_title = paper_data.get("Title", "").lower()
            if paper_title:
                # Take best match among all query titles
                title_score = max(
                    fuzz.token_sort_ratio(q_title.lower(), paper_title) / 100.0 
                    for q_title in q_titles
                )
                total_score += weights["titles"] * title_score
            max_possible_score += weights["titles"] 
        
        # --- Fuzzy match topics against summary ---
        if q_topics:
            paper_summary = paper_data.get("Title", "").lower() + " " + paper_data.get("Summary", "").lower()
            if paper_summary:
                # Check each topic against the full summary text
                for q_topic in q_topics:
                    topic_score = fuzz.partial_ratio(q_topic.lower(), paper_summary) / 100.0
                    total_score += weights["topics"] * topic_score
            max_possible_score += weights["topics"] * len(q_topics)
        
        # --- Fuzzy match authors ---
        if q_authors:
            paper_authors = paper_data.get("Authors", [])
            if paper_authors:
                # For each query author, find best match among paper authors
                for q_author in q_authors:
                    author_score = max(
                        fuzz.token_sort_ratio(q_author.lower(), a.lower()) / 100.0
                        for a in paper_authors
                    ) if paper_authors else 0.0
                    total_score += weights["authors"] * author_score
            max_possible_score += weights["authors"] * len(q_authors)
        
        # --- Match publication years (exact) ---
        if q_years:
            paper_published = paper_data.get("Published", "")
            paper_year = paper_published[:4] if len(paper_published) >= 4 else ""
            if paper_year in q_years:
                total_score += weights["publicationYears"]
            max_possible_score += weights["publicationYears"] * len(q_years)
        
        # Normalize score to 0-1 range
        normalized_score = total_score / max_possible_score if max_possible_score > 0 else 0.0
        
        # Only keep papers above threshold
        if normalized_score >= MIN_THRESHOLD:
            scores[paper_id] = normalized_score
    # Fallback if no matches
    if not scores:
        return [] 
    breakpoint()
    # Sort by score descending and return top_n
    top_papers = sorted(scores, key=lambda pid: scores[pid], reverse=True)[:top_n]
    return top_papers
if __name__ == "__main__":

    paper_metadata_path = Path("vectorstores/demo_user/paper_metadata.json")
     
    with open(paper_metadata_path, "r") as f:
        metadata_json = json.load(f)

    # === Example query hints extracted from the LLM ===
    query_hints = {
        "titles": [""],
        "authors": [""],
        "topics": [""],
        "publicationYears": []
    }

    # === Run the filtering function ===
    top_papers = FuzzyMagic(metadata_json, query_hints, top_n=4)

    # === Print the result ===
    print("Top papers by fuzzy matching:", top_papers)