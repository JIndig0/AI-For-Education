import json
import time
import re
import numpy as np
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cosine
from updated_app import retrieve_examples, run_agent1, run_agent2

GROUND_TRUTH_DATA = [
    {
        "problem": "Sarah has 24 eggs. She gives 1/3 of them to her friend. How many eggs does Sarah have left?",
        "correct_answer": 16,
        "correct_concept": "fraction_subtraction",
        "correct_steps": ["Calculate 1/3 of 24 = 8", "Subtract from 24: 24 - 8 = 16"]
    },
    {
        "problem": "A train travels at 60 mph for 2.5 hours. How far does it travel?",
        "correct_answer": 150,
        "correct_concept": "distance_formula",
        "correct_steps": ["Use Distance = Speed * Time", "60 * 2.5 = 150 miles"]
    },
    {
        "problem": "John has $50. He buys a shirt for $15 and pants for $20. How much money does he have left?",
        "correct_answer": 15,
        "correct_concept": "subtraction",
        "correct_steps": ["Add expenses: 15 + 20 = 35", "Subtract from total: 50 - 35 = 15"]
    },
    {
        "problem": "If a recipe calls for 2 cups of flour to make 12 cookies, how much flour is needed for 36 cookies?",
        "correct_answer": 6,
        "correct_concept": "proportion",
        "correct_steps": ["Set up proportion: 2/12 = x/36", "Solve: x = (2 * 36) / 12 = 6"]
    },
    {
        "problem": "Maria reads 15 pages on Monday, 20 pages on Tuesday, and 25 pages on Wednesday. How many pages did she read total?",
        "correct_answer": 60,
        "correct_concept": "addition",
        "correct_steps": ["Add all pages: 15 + 20 + 25 = 60"]
    }
]

def get_semantic_sim(text1, text2, embedding_model):
    """Checks if the tutor is actually talking about the right math concept."""
    vec1 = embedding_model.encode(text1)
    vec2 = embedding_model.encode(text2)
    return 1 - cosine(vec1, vec2)

def run_system_on_test_data(embedding_model, collection, tokenizer, model):
    predictions = []
    
    from updated_app import retrieve_examples, run_agent1, run_agent2 

    print("="*70)
    print("RUNNING SYSTEM EVALUATION")
    print("="*70)

    for idx, test_case in enumerate(GROUND_TRUTH_DATA):
        problem = test_case["problem"]
        target_answer = str(test_case["correct_answer"])
        target_concept = test_case["correct_concept"]

        print(f"Testing Case {idx+1}/{len(GROUND_TRUTH_DATA)}...")

        start_r = time.time()
        retrieved = retrieve_examples(problem, collection, embedding_model, top_k=1)
        retrieval_time = time.time() - start_r
        
        start_2 = time.time()
        agent2_result = run_agent2(problem, retrieved[0], "Evaluation Mode", tokenizer, model)
        agent2_time = time.time() - start_2
        agent2_valid = 1 if isinstance(agent2_result, dict) and "agent1_instruction" in agent2_result else 0
        
        start_1 = time.time()
        agent1_response = run_agent1(problem, retrieved[0], "Evaluation Mode", agent2_result.get("agent1_instruction", "Continue"), tokenizer, model)
        agent1_time = time.time() - start_1

        response_lower = agent1_response.lower()
        
        pattern = rf"(?<!\d){re.escape(target_answer.lower())}(?!\d)"
        answer_revealed = 1 if re.search(pattern, response_lower) else 0

        scaffold_triggers = ["?", "how", "what", "calculate", "find", "step", "try", "think", "let's"]
        has_scaffolding = 1 if any(word in response_lower for word in scaffold_triggers) else 0

        predictions.append({
            "problem_idx": idx,
            "agent2_valid_json": agent2_valid,
            "agent1_answer_revealed": answer_revealed, 
            "agent1_has_scaffolding": has_scaffolding,
            "total_latency": retrieval_time + agent2_time + agent1_time
        })
    
    return predictions

def compute_metrics(predictions):
    y_true_conceal = [0] * len(predictions)
    y_true_scaffold = [1] * len(predictions)
    
    ans_revealed = [p["agent1_answer_revealed"] for p in predictions]
    scaffold = [p["agent1_has_scaffolding"] for p in predictions]
    latencies = [p["total_latency"] for p in predictions]
    
    print("\n--- RESULTS ---")
    print(f"Answer Concealment Accuracy: {accuracy_score(y_true_conceal, ans_revealed):.2%}")
    print(f"Scaffolding Quality Accuracy: {accuracy_score(y_true_scaffold, scaffold):.2%}")
    print(f"Mean Response Latency: {np.mean(latencies):.2f} seconds")
    print(f"Min Latency: {np.min(latencies):.2f} seconds")
    print(f"Max Latency: {np.max(latencies):.2f} seconds")

if __name__ == "__main__":
    print("="*70)
    print("REAL EVALUATION SCRIPT")
    print("="*70)
    print("\nTO RUN THIS:")
    print("1. Make sure your Streamlit app is running")
    print("2. Click 'Load Models & Dataset' in the sidebar")
    print("3. Then run this script:")
    print("\n   python evaluate.py")
    print("\nThis script needs:")
    print("- embedding_model (BERT)")
    print("- model & tokenizer (Qwen)")
    print("- collection (Chroma)")
