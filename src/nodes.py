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
    found_errors = [] #empty list to store failed tool calls, kept separate from real sources
    seen_content = set()

    for question in plan:
        print(f"[Researcher] Looking up: {question}")

        doc_answer = retrieve_documents.invoke(question) #use retrieve_document tool
        if doc_answer.startswith("Document retrieval failed"): #tool returned an error string, not real content
            found_errors.append(doc_answer) #log it instead of treating it as a real source
        elif doc_answer not in seen_content: #skip if we already saw this exact result
            seen_content.add(doc_answer)
            found_sources.append({"tool": "documents", "content": doc_answer}) #append the result to found_sources list

        web_answer = web_search.invoke(question) #use web_search tool
        if web_answer.startswith("Web search failed"): #same check for web tool failures
            found_errors.append(web_answer)
        elif web_answer not in seen_content:
            seen_content.add(web_answer)
            found_sources.append({"tool": "web", "content": web_answer})

    loops_so_far = state.get("iteration_count", 0)
    return {"sources": found_sources, "errors": found_errors, "iteration_count": loops_so_far + 1} #pass errors along too

#decide whether to continue search
def reflect_node(state):
    loops_so_far = state["iteration_count"]

    if loops_so_far >= MAX_LOOPS:
        print(f"[Reflect] Reached max loops ({MAX_LOOPS}). Stopping research.")
        return {"research_complete": True} #corrected boolean key to match graph.py routing function

    print(f"[Reflect] Loops so far: {loops_so_far}. Continuing research...")
    return {"research_complete": False}


#compile all sources into final answer
def responder_node(state):
    topic = state["topic"]
    sources = state["sources"]
    errors = state.get("errors", []) #any tool failures logged by researcher_node

    all_info = "\n\n".join(s["content"] for s in sources)

    prompt = f"Using only this information, write a clear answer to: {topic}\n\nInformation:\n{all_info}\n\nInclude a Sources section at the end. List each unique source ONLY ONCE, even if it appears multiple times above."
    response = llm.invoke(prompt) #prompt llm for final answer
    final_answer = response.content

    if errors: #let the user know something failed instead of hiding it
        final_answer += "\n\n(Note: some sources could not be retrieved: " + "; ".join(errors) + ")"

    print("[Responder] Done.")

    return {
        "final_answer": final_answer,
        "messages": [AIMessage(content=final_answer)],
    }