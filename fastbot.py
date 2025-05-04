import csv
import re
import multiprocessing
from multiprocessing import Pool
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import requests
import os
import time
import json
import google.generativeai as genai

# --- MUST BE FIRST: Streamlit page config ---
st.set_page_config(
    page_title="🎓 College Info Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inject responsive viewport meta tag
st.markdown("""
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
""", unsafe_allow_html=True)

# --- Centered Title and Subtitle ---
st.markdown("""
    <style>
    .centered-title {
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 0.2em;
    }
    .centered-subtitle {
        text-align: center;
        font-size: 1.2em;
        color: #555;
        margin-bottom: 1em;
    }
    </style>

    <div class="centered-title">🎓 College Info Assistant</div>
    <div class="centered-subtitle">
        An Intelligent Chatbot for College Search Powered by FAISS, Sentence Transformers, and Gemini Flash
        <br>ALL MESSAGES ARE CURRENTLY MONITORED BY ADMIN FOR IMPROVING CHATBOT PERFORMANCE
    </div>
    <hr>
""", unsafe_allow_html=True)

# --- Configuration ---
CSV_FILE = 'standardized_finatdata.csv'
TXT_FILE = 'institution_descriptions.txt'

# --- Gemini Model Setup ---
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
llm_model = genai.GenerativeModel('gemini-2.0-flash')
# --- Utility Functions ---
def clean_field_name(field_name):
    field_name = field_name.replace('_', ' ').replace('\n', ' ').strip().capitalize()
    field_name = re.sub(' +', ' ', field_name)
    return field_name



def process_row(row):
    description = ""
    institution_name = row.get('name_of_the_institution_full_name', '').strip()
    if institution_name:
        description += f"{institution_name}."
    else:
        description += "Institution Name: Not Available."

    for field_name, field_value in row.items():
        if not field_value:
            continue
        field_value = field_value.strip()
        if field_value.lower() in ['n', 'no', 'Nil']:
            continue
        if field_name != 'Institution_Name':
            clean_name = clean_field_name(field_name)
            description += f" {clean_name}: {field_value}."

    return description.strip()

def generate_metadata_from_csv(csv_filepath, output_txt_path, num_workers=None):
    if os.path.exists(output_txt_path):
        return
    with open(csv_filepath, 'r', encoding='utf-8') as csvfile:
        reader = list(csv.DictReader(csvfile))
    with Pool(processes=num_workers or multiprocessing.cpu_count()) as pool:
        paragraphs = pool.map(process_row, reader)
    with open(output_txt_path, 'w', encoding='utf-8') as outfile:
        for paragraph in paragraphs:
            outfile.write(paragraph + '\n' + '-' * 40 + '\n')

@st.cache_resource
def load_data_and_embeddings():
    with open(TXT_FILE, 'r', encoding='utf-8') as file:
        texts = file.read().split('----------------------------------------')
    texts = [text.strip() for text in texts if text.strip()]
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return embedding_model, texts, index



def retrieve_relevant_context(query, top_k, distance_threshold=0.7):
    query_emb = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_emb), top_k)
    context = []
    for dist, idx in zip(distances[0], indices[0]):
        if dist < distance_threshold:
            context.append(texts[idx])
    return "\n\n".join(context)


def ask_gemini(context, question):
    prompt = f"""
You are a helpful, intelligent assistant that provides concise, accurate answers about colleges.

Only include relevant, available information. Omit fields that are missing or marked 'Nil'. Use clear, professional language.

### CONTEXT:
{context}

### QUESTION:
{question}

### INSTRUCTIONS:
- Answer only what the user asked.
- Omit unrelated or unavailable details.
- Expand abbreviations (e.g., BSc → Bachelor of Science).
- If asked about a specific field , list all the names in it.
- Never say "data not available".
- Be precise, use complete sentences.

"""
    print("\n\n--- FINAL PROMPT TO GEMINI ---\n")
    print(prompt)
    print("\n--- END PROMPT ---\n")
    try:
        response = llm_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Gemini error: {e}"

# --- Memory Persistence ---
MEMORY_FILE = "chat_memory.json"

def save_memory():
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state["messages"], f)

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            st.session_state["messages"] = json.load(f)

# --- Main App Logic ---
generate_metadata_from_csv(CSV_FILE, TXT_FILE)
embedding_model, texts, index = load_data_and_embeddings()
TOP_K = len(texts)

if "messages" not in st.session_state:
    st.session_state["messages"] = []
    load_memory()

if not st.session_state["messages"]:
    welcome_message = "👋 Hello! How can I help you today? I can assist you with any college information you need."
    st.session_state["messages"].append({"role": "assistant", "content": welcome_message})
    save_memory()

# --- Sidebar: Chat History ---
with st.sidebar:
    st.header("🕑 Chat History")
    if st.session_state["messages"]:
        for msg in st.session_state["messages"]:
            st.markdown(f"**{msg['role'].capitalize()}**: {msg['content'][:30]}...")
    else:
        st.markdown("*No chats yet.*")
    if st.button("🧹 Clear Chat"):
        st.session_state["messages"] = []
        save_memory()
        st.rerun()
    if st.button("📥 Download Chat"):
        if st.session_state["messages"]:
            chat_text = "\n\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state["messages"]])
            st.download_button("Download as TXT", data=chat_text, file_name="chat_history.txt", mime="text/plain")

# --- Chat Interface ---
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(f"<div class='chat-bubble'>{msg['content']}</div>", unsafe_allow_html=True)

user_query = st.chat_input("Type your question here...")

if user_query:
    st.session_state["messages"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(f"<div class='chat-bubble'>{user_query}</div>", unsafe_allow_html=True)
    with st.spinner("Thinking..."):
        context = retrieve_relevant_context(user_query, TOP_K)
        raw_answer = ask_gemini(context, user_query)
    final_answer = ""
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        for i in range(len(raw_answer)):
            final_answer = raw_answer[:i+1]
            answer_placeholder.markdown(f"<div class='chat-bubble'>{final_answer}</div>", unsafe_allow_html=True)
            time.sleep(0.01)
    st.session_state["messages"].append({"role": "assistant", "content": raw_answer})
    save_memory()
