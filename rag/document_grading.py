import asyncio
import logging
from config import llm
from core.schemas import State, DocRelevance
logger = logging.getLogger(__name__)

async def grade_docs(state: State):
    logger.info("Launching parallel workers for individual document grading...")
    
    docs = state.get("retrievedDocs", [])
    question = state.get("rewrittenQuestion") or state.get("originalQuestion")
    
    if not docs:
        return {"relevancePassed": False, "unanswered": question}

    # structured grader
    grader_llm = llm.with_structured_output(DocRelevance)

    async def check_doc(doc):
        prompt = f"Question: {question}\n\nDocument: {doc.page_content}"
        result = await grader_llm.ainvoke(prompt)
        return doc, result

    # Launch checks in parallel
    tasks = [check_doc(d) for d in docs]
    results = await asyncio.gather(*tasks)

    # Keep only documents that are relevant to the question (fully and partially)
    filtered_docs = [
        doc for doc, report in results 
        if report.grade != "completely irrelevant"
    ]
    
    logger.info(f"Filtered {len(docs)} down to {len(filtered_docs)} relevant documents.")
    return {"retrievedDocs": filtered_docs}
