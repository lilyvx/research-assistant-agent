from src.graph import graph

config = {"configurable": {"thread_id": "test-1"}}

result = graph.invoke({"topic": "multi-head attention"}, config)

print("\n=== FINAL ANSWER ===")
print(result["final_answer"])