import json
import os
import requests
import random
import numpy as np
import faiss

MEMORY_FILE = "memory.json"

# Helpers

def get_embedding(text):
    url = "http://localhost:11434/api/embeddings"
    data = {"model": "gemma3:1b", "prompt": text}
    try:
        r = requests.post(url, json=data)
        return r.json().get("embedding", []) if r.status_code == 200 else []
    except Exception as e:
        print(f"[Embedding error: {e}]")
        return []

def save_memories(memories):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memories, f, ensure_ascii=False, indent=2)

def load_memories():
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    migrated = []
    changed = False
    for m in data:
        if isinstance(m, dict):
            emb = get_embedding(m["input"])
            if emb:
                migrated.append([emb, m["input"], m["response"], m.get("feedbackRating", 1.0)])
                changed = True
        elif isinstance(m, list) and len(m) == 4 and isinstance(m[0], list) and m[0]:
            migrated.append(m)
    if changed:
        save_memories(migrated)
    return migrated

def build_faiss_index(memories):
    valid = [m for m in memories if isinstance(m, list) and len(m) == 4 and isinstance(m[0], list) and m[0]]
    if not valid:
        return None, None
    dim = len(valid[0][0])
    index = faiss.IndexFlatL2(dim)
    vectors = np.array([m[0] for m in valid]).astype('float32')
    index.add(vectors)
    return index, vectors

def search_memory(index, memories, query_emb, top_k=1):
    if not index or not memories or not query_emb:
        return None, None
    D, I = index.search(np.array([query_emb]).astype('float32'), top_k)
    idx = I[0][0]
    return memories[idx][1], idx

def ask_gemma_question():
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "gemma3:1b", "prompt": "Ask a new question to learn more about the user.", "stream": False}
    )
    return r.json().get("response", "")

print("Liya is learning. Talk to it:")
asked_questions = set()
memories = load_memories()
index, vectors = build_faiss_index(memories) if memories else (None, None)

def handle_question(question):
    print(f"Liya asks you (from memory): {question}")
    input("Your answer: ")
    asked_questions.add(question)
    for m in memories:
        if m[1] == question:
            m[3] = 1.0
    save_memories(memories)

def add_memory(q, a):
    emb = get_embedding(q)
    memories.append([emb, q, a, 1.0])
    save_memories(memories)
    return build_faiss_index(memories)

# Initial question
if memories:
    qs = [m[1] for m in memories if m[1] not in asked_questions]
    if qs:
        handle_question(random.choice(qs))
else:
    q = ask_gemma_question()
    print("Gemma3 generated question:", q)
    a = input("Your answer: ").strip()
    index, vectors = add_memory(q, a)
    asked_questions.add(q)

while True:
    user_input = input("You: ").strip().lower()
    if user_input == "exit":
        break
    query_emb = get_embedding(user_input)
    found, idx = search_memory(index, memories, query_emb)
    if found:
        print("AI:", memories[idx][2])
        rating = 1.0 if input("Was this good? (y/n): ").strip().lower() == "y" else 0.0
        memories[idx][3] = rating
        save_memories(memories)
    else:
        response = input("AI doesn't know. What should it say?: ").strip()
        index, vectors = add_memory(user_input, response)
        print("AI learned that!")
    # Ask a question from memory or generate new
    qs = [m[1] for m in memories if m[1] not in asked_questions]
    if qs:
        handle_question(random.choice(qs))
    else:
        q = ask_gemma_question()
        print("Gemma3 generated question:", q)
        a = input("Your answer: ").strip()
        index, vectors = add_memory(q, a)
        asked_questions.add(q)
