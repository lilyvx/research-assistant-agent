"""
4 nodes:
  planner_node:turns the topic into a few search questions
  researcher_node: calls both tools to gather information
  reflect_node: decides enough info yet, or research more?
  responder_node: writes the final answer using everything gathered

each func takes state (shared) and returns a small
dict of what it want to update in that state.
"""

from langchain_core.messages import AIMessage
from src.llm import llm
from src.tools import retrieve_documents, web_search

MAX_LOOPS = 2  #search count 


def planner_node(state):
    topic = state["topic"] 

    prompt = f"Break this topic into 3 short search questions, one per line, no numbering:\n{topic}" #simple prompt
    response = llm.invoke(prompt) 

    questions = [line.strip() for line in response.content.split("\n") if line.strip()] #parse response into list of questions

    print(f"[Planner] {len(questions)} questions:", questions)

    return {"plan": questions} #returns a dict with plan key(a list of questions for researcher node to use)

