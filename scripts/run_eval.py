import json
from src.parsing.parser import PythonCodeParser
from src.graph.builder import DependencyGraphBuilder

def run_evaluation():
    print("Loading CodeGraphEval sample dataset...")
    with open("data/CodeGraphEval_50_sample.json", "r") as f:
        eval_cases = json.load(f)

    print(f"Loaded {len(eval_cases)} cases. Initializing dummy Agent...\n")
    
    # Initialize Core Infrastructure
    parser = PythonCodeParser()
    graph = DependencyGraphBuilder()
    
    # Load dummy graph data matching the eval use case
    graph.build_dummy_graph()

    score = 0
    
    for case in eval_cases:
        question = case["question"]
        expected = case["ground_truth_answer"]
        print(f"Q: {question}")
        
        # MOCK AGENT LOGIC (Hardcoded exact-match for the demonstration)
        if "authenticate_user function" in question:
            # Agent decides to use graph tools
            callers = graph.get_callers("authenticate_user")
            # Agent formulates an answer based on graph
            agent_answer = f"The authenticate_user function is called by the {callers[0]} in auth.py."
            agent_evidence = [{"filepath": "auth.py", "line_ranges": [12, 15]}]
            
        elif "timeout configuration" in question:
            # Agent decides to use retrieval tools
            agent_answer = "The default timeout is defined as 30 seconds in config.py."
            agent_evidence = [{"filepath": "config.py", "line_ranges": [4, 4]}]
            
        else:
            agent_answer = "I don't know."
            agent_evidence = []

        print(f"Expected: {expected}")
        print(f"Agent answered: {agent_answer}")
        
        if agent_answer == expected:
            print("[PASS] Exact match!")
            score += 1
        else:
            print("[FAIL] Mismatch")
        print("-" * 40)
        
    print(f"\nFinal Score: {score}/{len(eval_cases)} ({score/len(eval_cases)*100}%)")

if __name__ == "__main__":
    run_evaluation()
