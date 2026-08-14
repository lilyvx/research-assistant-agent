from src.nodes import planner_node, researcher_node, reflect_node, responder_node

#test planner_node
fake_state = {"topic": "multi-head attention "}

result = planner_node(fake_state)
print(result)


#test researcher_node
fake_state = {
    "plan": [
        "What is multi-head attention?",
        "How does multi-head attention differ from single-head attention?",
    ],
    "iteration_count": 0,
}

result = researcher_node(fake_state)
print(result)

#test reflect_node
fake_state = {
    "topic": "",
    "sources": [
         {"tool": "documents", "content": "Multi-head attention allows the model to jointly attend to information from different representation subspaces."},
        {"tool": "web", "content": "Multi-head attention runs several attention mechanisms in parallel."},
    ],
    "iteration_count": 1,
}

result = reflect_node(fake_state)
print(result)



#test responder_node
from src.nodes import responder_node

fake_state = {
    "topic": "multi-head attention",
    "sources": [
        {"tool": "documents", "content": "Multi-head attention allows the model to jointly attend to information from different representation subspaces."},
        {"tool": "web", "content": "Multi-head attention runs several attention mechanisms in parallel."},
    ],
}

result = responder_node(fake_state)
print(result["final_answer"])