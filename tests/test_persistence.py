from src.graph import graph

#same thread_id used for both calls 
config = {"configurable": {"thread_id": "persistence-test-1"}}

print("=== Call 1: first topic ===")
result_1 = graph.invoke({"topic": "multi-head attention"}, config)
print(f"Number of messages after call 1: {len(result_1['messages'])}")
for m in result_1["messages"]:
    print(f"  - {type(m).__name__}: {m.content[:60]}...")

print("\n=== Call 2: second topic, SAME thread_id ===")
result_2 = graph.invoke({"topic": "positional encoding in transformers"}, config)
print(f"Number of messages after call 2: {len(result_2['messages'])}")
for m in result_2["messages"]:
    print(f"  - {type(m).__name__}: {m.content[:60]}...")

print("\n=== Verdict ===")
if len(result_2["messages"]) > len(result_1["messages"]):
    print("PASS: message count grew across calls, persistence is working.")
else:
    print("FAIL: message count did not grow,state is not persisting.")