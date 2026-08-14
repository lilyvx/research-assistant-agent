"""
graph.py:
connect the four node from nodespy into a LangGraph
StateGraph: defines steps and the conditional loop
between researcher and reflect,
attaches a checkpointer for coversation memory
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.state import AgentState
from src.nodes import planner_node, researcher_node, reflect_node, responder_node

#routing func langraph calls after reflect node runs
def route_after_reflect(state: AgentState) -> str:

    if state["research_complete"]:
        return "respond"
    else:
        return "continue_research"

#graph builder
graph = StateGraph(AgentState)

#register node functions under name for edge reference 
graph.add_node("planner", planner_node)
graph.add_node("researcher", researcher_node)
graph.add_node("reflect", reflect_node)
graph.add_node("responder", responder_node)

#edge (fixed)
graph.add_edge(START, "planner")
graph.add_edge("planner", "researcher")
graph.add_edge("researcher", "reflect")
graph.add_edge("responder", END)

#conditional edge, call route_after_reflect after reflect_node runs to decide where to go next
#the dict map the string that function returns to the actual node name to jump to
graph.add_conditional_edges(
    "reflect",
    route_after_reflect,
    {
        "continue_research": "researcher", 
        "respond": "responder",
    }, 
)

#attach checkpointer to persist across different graph.invoke
#calls that share the same thread_id to give multi turn conversation memory
checkpointer = MemorySaver()

graph = graph.compile(checkpointer=checkpointer)