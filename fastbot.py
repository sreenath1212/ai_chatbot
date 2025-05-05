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
from difflib import get_close_matches

# --- Streamlit config ---
st.set_page_config(
    page_title="🎓 IHRD InfoBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
""", unsafe_allow_html=True)



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

    <div class="centered-title">🎓 IHRD InfoBot</div>
    <div class="centered-subtitle">
        An Intelligent Chatbot for IHRD College Informations
        <br>Powered by FAISS, Sentence Transformers, and  Google-GenerativeAi(Gemini Flash)
    </div>
    <hr>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        color: #999;
        text-align: center;
        font-size: 0.85em;
        padding: 10px 0;
        border-top: 1px solid #eee;
        z-index: 100;
    }
    
    
    </style>
    <div class="footer">
        © 2025 IHRD InfoBot | Built with ❤️ by COLLEGE OF APPLIED SCIENCE MAVELIKKARA
    </div>
    """,
    unsafe_allow_html=True
)

# --- Configuration ---
CSV_FILE = 'standardized_finatdata.csv'
MEMORY_FILE = "chat_memory.json"

# --- Gemini Setup ---
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
llm_model = genai.GenerativeModel('gemini-2.0-flash')

# --- Utilities ---
def clean_field_name(field_name):
    field_name = field_name.replace('_', ' ').replace('\n', ' ').strip().capitalize()
    field_name = re.sub(' +', ' ', field_name)
    return field_name

def get_all_field_names(csv_path):
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return [clean_field_name(field) for field in reader.fieldnames]

def match_fields_from_query(query, field_names, cutoff=0.5):
    query_lower = query.lower()
    matches = [f for f in field_names if any(word in f.lower() for word in query_lower.split())]
    if not matches:
        matches = get_close_matches(query_lower, field_names, n=3, cutoff=cutoff)
    return matches

def process_row(row):
    data = {}
    institution_name = row.get('name_of_the_institution_full_name', '').strip()
    data["Institution Name"] = institution_name if institution_name else "Not Available"
    for field_name, field_value in row.items():
        if field_value and field_value.lower() not in ['n', 'no', 'nil']:
            clean_name = clean_field_name(field_name)
            data[clean_name] = field_value.strip()
    return data

@st.cache_resource
def load_data_and_embeddings():
    with open(CSV_FILE, 'r', encoding='utf-8') as csvfile:
        reader = list(csv.DictReader(csvfile))
        processed_data = [process_row(row) for row in reader]

    texts = [" ".join(f"{k}: {v}" for k, v in item.items()) for item in processed_data]
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return embedding_model, processed_data, texts, index

def is_course_query(query):
    course_keywords = ['bsc', 'msc', 'btech', 'mtech', 'ba', 'ma', 'bcom', 'mcom', 'mba', 'bca', 'dca', 'pgdca', 'bba', 'dvoc', 'btm', 'mca', 'computer', 'science', 'electronics', 'engineering', 'commerce', 'business', 'administration', 'arts', 'journalism', 'literature', 'taxation', 'finance', 'accounting', 'logistics', 'supply chain', 'co-operation', 'applications', 'management', 'data', 'analytics', 'ai', 'ml', 'cyber', 'security', 'forensics', 'information', 'vlsi', 'embedded', 'systems', 'energy', 'biomedical', 'electrical', 'mechanical', 'automobile', 'civil', 'robotics', 'automation', 'hardware', 'technology', 'design', 'science']
    return any(kw in query.lower() for kw in course_keywords)

def get_course_matches_from_csv(query):
    matched = []
    course_keywords = query.lower().split()
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ug = row.get('list_of_ug_courses_and_intake', '').lower()
            pg = row.get('list_of_pg_courses_and_intake', '').lower()
            if any(kw in ug or kw in pg for kw in course_keywords):
                matched.append(row)
    return matched

def build_context_for_course_query(query):
    results = get_course_matches_from_csv(query)
    paragraphs = []
    for row in results:
        inst = row.get('name_of_the_institution_full_name', 'Institution').strip()
        ug = row.get('list_of_ug_courses_and_intake', '').strip()
        pg = row.get('list_of_pg_courses_and_intake', '').strip()
        block = f"Institution: {inst}."
        if ug:
            block += f" UG Courses: {ug}."
        if pg:
            block += f" PG Courses: {pg}."
        paragraphs.append(block)
    return "\n\n".join(paragraphs)

def retrieve_filtered_context(query, top_k, field_names, processed_data):
    if is_course_query(query):
        return build_context_for_course_query(query)
    matched_fields = match_fields_from_query(query, field_names)
    query_emb = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_emb), top_k)

    filtered_context = []
    for i in indices[0]:
        row = processed_data[i]
        context_lines = [f"Institution: {row.get('Institution Name', 'Unknown')}"]
        for field in matched_fields:
            value = row.get(field)
            if value:
                context_lines.append(f"{field}: {value}")
        filtered_context.append("\n".join(context_lines))

    return "\n\n".join(filtered_context)

def ask_gemini(context, question):
    history = ""
    for msg in st.session_state["messages"][-4:]:
        if msg["role"] == "user":
            history += f"\nUser: {msg['content']}"
        elif msg["role"] == "assistant":
            history += f"\nAssistant: {msg['content']}"

    prompt = f"""
You are a knowledgeable and friendly IHRD college information assistant. Your job is to provide clear, helpful, and human-like answers using the given data.

Use information provided in the context below. If the user is asking about IHRD institutions in general, you may use external knowledge for general facts like locations other details etc.

Use the recent chat history to understand follow-up questions or references to previous answers.

Your responses should sound natural and conversational — don't just list data, explain it briefly like a human would.

Do not tell about internal process, data and all. Act like a normal human.

### CONTEXT:
{context}

### CHAT HISTORY:
{history}

### USER QUESTION:
{question}

### IMPORTANT INSTRUCTIONS:
- Only include fields specifically asked about. Do not include unrelated details.
- Treat these as distinct categories: research center, funded projects, industry-on-campus initiatives, earn-while-you-learn, incubation centers, startup initiatives, skill centers, MoUs, and internships.
- Do not mention or hint at missing or unavailable data — just skip it.
- If the question mentions a specific course (e.g., "MSc CS", "BSc Physics", "BTech in Electronics"), return:
  - Only the **institution name** (`name_of_the_institution_full_name`),
  - And the course fields:
    - `list_of_pg_courses_and_intake` if it's a PG course (like MSc, MCom, MA),
    - `list_of_ug_courses_and_intake` if it's a UG course (like BSc, BA, BCom, BTech),
    - Or both if unsure.
  - Do NOT include any other fields even if they are available.
- Expand abbreviations (e.g., "BSc" → "Bachelor of Science", "IHRD", "MVK","TVM") where appropriate, using your general knowledge.
- If a user asks about a list (e.g., courses or projects), list only the relevant names or entries.
- Be detailed and professional, but keep a friendly, natural tone.
- Never say "data not available". Just focus on what *is* available and relevant.
"""
    print("\n\n--- FINAL PROMPT TO GEMINI ---\n")
    print(prompt)
    print("\n--- END PROMPT ---\n")
    try:
        response = llm_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Gemini error: {e}"

# --- Memory persistence ---
def save_memory():
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state["messages"], f)

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            st.session_state["messages"] = json.load(f)

# --- Main App Logic ---
embedding_model, processed_data, texts, index = load_data_and_embeddings()
all_field_names = get_all_field_names(CSV_FILE)
TOP_K = len(texts)

if "messages" not in st.session_state:
    st.session_state["messages"] = []
    load_memory()

if not st.session_state["messages"]:
    welcome_message = "👋 Hello! How can I help you today? I can assist you with any college information you need."
    st.session_state["messages"].append({"role": "assistant", "content": welcome_message})
    save_memory()

# --- Sidebar ---
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

# --- Chat Display ---
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(f"<div class='chat-bubble'>{msg['content']}</div>", unsafe_allow_html=True)

user_query = st.chat_input("Type your question here...")

if user_query:
    st.session_state["messages"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(f"<div class='chat-bubble'>{user_query}</div>", unsafe_allow_html=True)
    with st.spinner("Typing..."):
        context = retrieve_filtered_context(user_query, top_k=TOP_K, field_names=all_field_names, processed_data=processed_data)
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
