"""
4 nodes:
  planner_node:turns the topic into a few search questions
  researcher_node: calls 2 tools frm toolpy to gather information
  reflect_node: decides enough info yet or research more?
  responder_node: writes the final answer using everything gathered

each func takes state (shared) and returns a small
dict of what it want to update in that state
"""

from langchain_core.messages import AIMessage
from src.llm import llm
from src.tools import retrieve_documents, web_search

MAX_LOOPS = 2  #search count 


def planner_node(state):
    topic = state["topic"]  #set at statepy

    prompt = f"Break this topic into 3 short search questions, one per line, no numbering:\n{topic}" #simple prompt
    response = llm.invoke(prompt) 

    questions = [line.strip() for line in response.content.split("\n") if line.strip()] #parse response into list of questions

    print(f"[Planner] {len(questions)} questions:", questions)

    return {"plan": questions} #returns a dict with plan key(a list of questions for researcher node to use)

#research with retrievedoc and websearch
def researcher_node(state):
    plan = state["plan"] #take subquestion
    found_sources = [] #empty list to store sources found by tools

    for question in plan:
        print(f"[Researcher] Looking up: {question}")

        doc_answer = retrieve_documents.invoke(question) #use retrieve_document tool 
        found_sources.append({"tool": "documents", "content": doc_answer}) #append the result to found_sources list

        web_answer = web_search.invoke(question)#use web_search tool
        found_sources.append({"tool": "web", "content": web_answer})

    loops_so_far = state.get("iteration_count", 0)

    return {
        "sources": found_sources,
        "iteration_count": loops_so_far + 1,
    }

#decide whether to continue search
def reflect_node(state):
    loops_so_far = state["iteration_count"]

    if loops_so_far >= MAX_LOOPS:
        print(f"[Reflect] Reached max loops ({MAX_LOOPS}). Stopping research.")
        return {"research_complete": True} #corrected boolean key to match graph.py routing function

    print(f"[Reflect] Loops so far: {loops_so_far}. Continuing research.")
    return {"research_complete": False}


#compile all sources into final answer
def responder_node(state):
    topic = state["topic"]
    sources = state["sources"]

    all_info = "\n\n".join(s["content"] for s in sources) 

    prompt = f"Using only this information, write a clear answer to: {topic}\n\nInformation:\n{all_info}\n\nInclude a Sources section at the end."
    response = llm.invoke(prompt) #prompt llm for final answer

    print("[Responder] Done.")

    return {
        "final_answer": response.content,
        "messages": [AIMessage(content=response.content)],
    }