from config import llm
from core.schemas import State
from core.prompts import get_casual_generation_prompt
from langchain_core.messages import HumanMessage, SystemMessage

async def handle_general_talk(state: State):
    
    conversation_summary = state.get("conversationSummary", "")
    system_prompt = get_casual_generation_prompt(conversation_summary)
    user_query = state.get("rewrittenQuestion") or state.get("originalQuestion")
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query)
    ])

    return {
        "messages": [response],          
        "finalAnswer": response.content       
    }
    