from core.schemas import State, RuntimeContext
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from rag import (
    summarize_conversation_history,
    analyze_query,
    fuzzy_match_papers,
    retrieve, 
    generate
)

# Initialize checkpointer
checkpointer = InMemorySaver()

# Build workflow graph
graph_builder = StateGraph(State, context_schema=RuntimeContext)

# Nodes
graph_builder.add_node("summarize_conv", summarize_conversation_history)
graph_builder.add_node("analyze_query", analyze_query)
graph_builder.add_node("scope_context", fuzzy_match_papers)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)

# Edges
graph_builder.add_edge(START, "summarize_conv")
graph_builder.add_edge("summarize_conv", "analyze_query")
graph_builder.add_edge("analyze_query", "scope_context")
graph_builder.add_edge("scope_context", "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)

# Compile graph
workflow = graph_builder.compile(
    checkpointer=checkpointer
)

img = workflow.get_graph(xray=True).draw_mermaid_png()
with open("assets/workflow.png", "wb") as f:
    f.write(img)

