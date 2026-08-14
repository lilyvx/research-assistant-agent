from unittest.mock import patch
from src.graph import graph

config = {"configurable": {"thread_id": "error-test-1"}}

#patch ddg itself (a plain class, not a Pydantic tool object) so
#that constructing it raises an exception simulating ddg
#being unreachable. must match where ddg is imported
#and used: inside src.tools
with patch("src.tools.DDGS", side_effect=Exception("Simulated network outage")):
    try:
        result = graph.invoke({"topic": "multi-head attention"}, config)
        print("=== Graph completed without crashing ===")
        print(f"\nFinal answer:\n{result['final_answer']}")
    except Exception as error:
        print("=== FAIL: graph crashed instead of degrading gracefully ===")
        print(f"Error: {error}")