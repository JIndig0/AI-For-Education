import streamlit as st
import pandas as pd
import json
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Page config
st.set_page_config(page_title="AI Math Tutor", layout="wide")

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "collection" not in st.session_state:
    st.session_state.collection = None
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
if "model" not in st.session_state:
    st.session_state.model = None

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_generation_model():
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return tokenizer, model

@st.cache_resource
def setup_chromadb(chunks, embeddings):
    client = chromadb.Client()
    collection = client.create_collection(name="mathdial", configuration={"hnsw": {"space": "cosine"}})
    
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    ids = [str(i) for i in range(len(chunks))]
    
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    return collection

def retrieve_examples(query, collection, embedding_model, top_k=5):
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    
    retrieved_questions = results["documents"][0]
    retrieved_metadata = results["metadatas"][0]
    
    examples = []
    for i in range(len(retrieved_questions)):
        example = {
            "question": retrieved_questions[i],
            "qid": retrieved_metadata[i]["qid"],
            "student_incorrect_solution": retrieved_metadata[i]["student_incorrect_solution"],
            "teacher_described_confusion": retrieved_metadata[i]["teacher_described_confusion"],
            "ground_truth": retrieved_metadata[i]["ground_truth"],
        }
        examples.append(example)
    
    return examples

def generate_text(messages, tokenizer, model, max_new_tokens=200):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()

def is_math_problem(problem, tokenizer, model):
    messages = [
        {"role": "system", "content": "Return ONLY valid JSON:\n{\"is_math_problem\": true/false}\nNo extra text."},
        {"role": "user", "content": problem}
    ]
    response = generate_text(messages, tokenizer, model, max_new_tokens=30)
    try:
        return json.loads(response).get("is_math_problem", False)
    except:
        return False

def run_agent2(problem, retrieved_example, history_text, tokenizer, model):
    messages = [
        {"role": "system", "content": """You are an evaluator. Compare student reasoning with ground truth.
Return ONLY valid JSON:
{"student_understanding": "none/partial/good", "missing_concept": "text", "agent1_instruction": "text"}"""},
        {"role": "user", "content": f"Problem:\n{problem}\n\nRetrieved:\n{retrieved_example['question']}\n\nGround truth:\n{retrieved_example['ground_truth']}\n\nHistory:\n{history_text}"}
    ]
    response = generate_text(messages, tokenizer, model, max_new_tokens=150)
    try:
        return json.loads(response)
    except:
        return {"student_understanding": "unknown", "missing_concept": "Error", "agent1_instruction": "Continue"}

def run_agent1(problem, retrieved_example, history_text, agent2_instruction, tokenizer, model):
    messages = [
        {"role": "system", "content": "You are a flawed math tutor. Give incorrect/partial responses based on retrieved examples."},
        {"role": "user", "content": f"Problem:\n{problem}\n\nRetrieved incorrect:\n{retrieved_example['student_incorrect_solution']}\n\nConfusion:\n{retrieved_example['teacher_described_confusion']}\n\nHistory:\n{history_text}\n\nInstruction:\n{agent2_instruction}"}
    ]
    return generate_text(messages, tokenizer, model, max_new_tokens=120)

def format_history(history):
    return "\n".join(f"{m['role']}: {m['content']}" for m in history)

# Sidebar for setup
st.sidebar.header("Setup")
if st.sidebar.button("Load Models & Dataset"):
    with st.spinner("Loading models..."):
        st.session_state.embedding_model = load_embedding_model()
        st.session_state.tokenizer, st.session_state.model = load_generation_model()
        
        with st.spinner("Loading dataset..."):
            ds = load_dataset("eth-nlped/mathdial")
            df = ds["train"].to_pandas()
            cols_required = ["teacher_described_confusion", "self-correctness", "self-typical-confusion", "self-typical-interactions"]
            df = df.dropna(subset=cols_required).reset_index(drop=True)
            cols_to_remove = ["scenario", "self-typical-interactions", "self-typical-confusion", "self-correctness"]
            df = df.drop(columns=cols_to_remove)
            df = df.drop_duplicates(subset="qid").reset_index(drop=True)
            
            chunks = [{"text": row["question"], "metadata": {"qid": row["qid"], "student_incorrect_solution": row["student_incorrect_solution"], "teacher_described_confusion": row["teacher_described_confusion"], "ground_truth": row["ground_truth"]}} for _, row in df.iterrows()]
            
            embeddings = st.session_state.embedding_model.encode([c["text"] for c in chunks], show_progress_bar=True, normalize_embeddings=True)
            st.session_state.collection = setup_chromadb(chunks, embeddings)
    
    st.sidebar.success("Ready!")

# Main app
st.title("🧮 AI Math Tutor")

if st.session_state.collection is None:
    st.warning("Please load models and dataset from the sidebar first.")
else:
    # Input section
    problem = st.text_area("Enter a math problem:", height=100)
    
    if st.button("Start Tutoring Session"):
        if problem:
            st.session_state.conversation_history = [{"role": "student", "content": problem}]
    
    # Display conversation
    if st.session_state.conversation_history:
        st.subheader("Conversation:")
        for msg in st.session_state.conversation_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        
        # Student response
        student_response = st.text_area("Your response:", key="student_input")
        
        if st.button("Send"):
            if student_response:
                st.session_state.conversation_history.append({"role": "student", "content": student_response})
                
                with st.spinner("Agent thinking..."):
                    history_text = format_history(st.session_state.conversation_history)
                    retrieved = retrieve_examples(problem, st.session_state.collection, st.session_state.embedding_model, top_k=1)
                    
                    if retrieved:
                        agent2_result = run_agent2(problem, retrieved[0], history_text, st.session_state.tokenizer, st.session_state.model)
                        agent1_reply = run_agent1(problem, retrieved[0], history_text, agent2_result["agent1_instruction"], st.session_state.tokenizer, st.session_state.model)
                        st.session_state.conversation_history.append({"role": "assistant", "content": agent1_reply})
                
                st.rerun()