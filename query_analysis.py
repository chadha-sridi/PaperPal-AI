import os
from config import llm
from schemas import State, QueryAnalysis
from prompts import get_query_analysis_prompt
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

def analyze_query(state: State) -> dict:
    last_user_msg = state["messages"][-1].content
    summary = state.get("conversationSummary", "")

    context = f"""
    Conversation summary:
    {summary}

    User question:
    {last_user_msg}
    """.strip()

    llm_structured = (
        llm
        .with_config(temperature=0.1)
        .with_structured_output(QueryAnalysis)
    )

    analysis: QueryAnalysis = llm_structured.invoke([
        SystemMessage(content=get_query_analysis_prompt()),
        HumanMessage(content=context)
    ])

    # Case 1: question is NOT clear → ask for clarification
    if not analysis.is_clear:
        return {
            "questionIsClear": False,
            "messages": [AIMessage(content=analysis.clarification_needed)]
        }

    # Case 2: question is clear → update state
    metadata_present = any(
        getattr(analysis.metadataHints, field)
        for field in analysis.metadataHints.model_fields
    )
    rewritten = analysis.rewrittenQuestion.strip() or last_user_msg
    return {
        "questionIsClear": True,
        "rewrittenQuestion": rewritten,
        "paperScope": analysis.paperScope,
        "metadataHintPresent": metadata_present,
        "metadataHints": analysis.metadataHints,
        "originalQuestion": state.get("originalQuestion") or last_user_msg
    }

if __name__ == "__main__":
    
    examples = [
    "Can you summarize the key contributions of the Attention Is All You Need paper?",
    "How do diffusion models compare to GANs for image generation in recent computer vision papers?",
    "What papers by Yann LeCun discuss self-supervised learning for vision tasks?",
    "Are there any arXiv papers after 2020 that combine graph neural networks with reinforcement learning?",
    "Which transformer-based models are most commonly used for clinical text classification?",
    "What differences exist between BERT and RoBERTa according to their original publications?",
    "How has causal inference been applied in healthcare machine learning research over the last few years?"
    ]
    for i, query in enumerate(examples, start=1):
        # Initialize a fresh state
        state = State(messages=[HumanMessage(content=query)])

        # Run query analysis
        updated_state = analyze_query(state)

        print(f"\n\n-------- Example {i} --------")
        print("Original Query:", query)
        print("Question Is Clear:", updated_state["questionIsClear"])
        print("Rewritten Question:", updated_state.get("rewrittenQuestion"))
        print("Paper Scope:", updated_state.get("paperScope"))
        print("Metadata Hint Present:", updated_state.get("metadataHintPresent"))
        if updated_state.get("metadataHints"):
            print("Metadata Hints:", updated_state["metadataHints"].model_dump())
        if not updated_state["questionIsClear"]:
            print("Clarification Needed:", updated_state["messages"][0].content)
