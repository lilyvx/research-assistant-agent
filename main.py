import uuid
from src.graph import graph


def main():

    print("=" * 60)
    print("Hi I am your Research Assistant Agent")
    print("Type a research topic to begin. Type 'x' to quit.")
    print("=" * 60)

    while True:
        topic = input("\nResearch topic: ").strip()

        if topic.lower() in ("exit", "quit", "x"):
            print("Goodbye!")
            break

        #evaluates to true when topic is an empty string
        if not topic:
            print("Please enter a topic, or type 'x' to quit.")
            continue

        print(f"\nResearching: {topic}\n") #interpolating the user's topic into a status message

        thread_id = str(uuid.uuid4()) #generates a random UUID object
        config = {"configurable": {"thread_id": thread_id}} #build dict with id 

        try:
            result = graph.invoke({"topic": topic}, config) #start the graph with the user topic and the config containing thread_id
        except Exception as error:
            print(f"Something went wrong while researching this topic: {error}")
            continue

        print("\n" + "=" * 30)
        print("ANSWER")
        print("=" * 30)
        print(result["final_answer"])
        print("=" * 60)


if __name__ == "__main__":
    main()