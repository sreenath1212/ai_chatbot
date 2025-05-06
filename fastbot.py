import csv
import re
import json
import os
import time
import ast
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# --- Streamlit config ---
st.set_page_config(
    page_title="\U0001F393 IHRD InfoBot",
    page_icon="\U0001F916",
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

    <div class="centered-title">\U0001F393 IHRD InfoBot</div>
    <div class="centered-subtitle">
        An Intelligent Chatbot for IHRD College Informations
        
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

# --- Constants ---
CSV_FILE = 'standardized_finatdata.csv'

# --- Primary Gemini ---
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
llm_model = genai.GenerativeModel('gemini-2.0-flash')

# --- Secondary Gemini for field identification ---
genai_field_match = genai.configure(api_key=st.secrets["SECOND_GEMINI_API_KEY"])
field_llm_model = genai.GenerativeModel('gemini-2.0-flash')

# --- Utility Functions ---
def clean_field_name(field_name):
    return re.sub(' +', ' ', field_name.replace('_', ' ').replace('\n', ' ').strip().capitalize())

def get_all_field_names(csv_path):
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return [clean_field_name(field) for field in reader.fieldnames]

def process_row(row):
    data = {}
    inst_name = row.get('name_of_the_institution_full_name', '').strip()
    data["Institution Name"] = inst_name if inst_name else "Not Available"
    for field_name, field_value in row.items():
        if field_value and field_value.lower() not in ['n', 'no', 'nil']:
            data[clean_field_name(field_name)] = field_value.strip()
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

def identify_fields_from_query_llm(query, all_fields):
    prompt = f"""
Given the user query:
"{query}"

Select the most relevant field names from the list below:
{json.dumps(all_fields, indent=2)}

Return ONLY a Python list of exact matching field names, nothing else. Example:
["Field A", "Field B"........."Field N"]


"""
    try:
        response = field_llm_model.generate_content(prompt)
        text = response.text.strip()

        match = re.search(r"\[.*?\]", text, re.DOTALL)
        if match:
            return ast.literal_eval(match.group(0))
        else:
            print("⚠️ No list detected in LLM response.")
            return []
    except Exception as e:
        print("Field LLM Error:", e)
        return []

def retrieve_filtered_context(query, top_k, field_names, processed_data):
    matched_fields = identify_fields_from_query_llm(query, field_names)
    query_emb = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_emb), top_k)
    output_blocks = []
    for i in indices[0]:
        row = processed_data[i]
        block = [f"Institution: {row.get('Institution Name', 'Unknown')}"]
        for field in matched_fields:
            value = row.get(field)
            if value:
                block.append(f"{field}: {value}")
        if len(block) > 1:
            output_blocks.append("\n".join(block))
    return "\n\n".join(output_blocks)

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

If the user asks about any of these courses -  ['bsc', 'msc', 'btech', 'mtech', 'ba', 'ma', 'bcom', 'mcom', 'mba', 'bca', 'dca', 'pgdca', 'bba', 'dvoc', 'btm', 'mca', 'computer', 'science', 'electronics', 'engineering', 'commerce', 'business', 'administration', 'arts', 'journalism', 'literature', 'taxation', 'finance', 'accounting', 'logistics', 'supply chain', 'co-operation', 'applications', 'management', 'data', 'analytics', 'ai', 'ml', 'cyber', 'security', 'forensics', 'information', 'vlsi', 'embedded', 'systems', 'energy', 'biomedical', 'electrical', 'mechanical', 'automobile', 'civil', 'robotics', 'automation', 'hardware', 'technology', 'design', 'science'], check the ug and pg field and return only details of courses user asked about.

Treat these as distinct categories: Applied Science Colleges, Engineering,Extension center, Technical Higher Secodary Schools, Regional Centre, Model Polytechnics, Study centre, and Model Finishing School and answer based on what user needs ,omit other categories even if available.

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
- Cas= applied science colleges only
  - Do NOT include any other fields even if they are available.
- Expand abbreviations (e.g., "BSc" → "Bachelor of Science", "IHRD", "MVK"-Mavelikkara,"TVM") where appropriate, using your general knowledge.
- If a user asks about a list (e.g., courses or projects), list only the relevant names or entries.
- Be detailed and professional, but keep a friendly, natural tone.
- Never say "data not available". Just focus on what *is* available and relevant.
"""
    try:
        response = llm_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Gemini error: {e}"

# --- Load Data ---
embedding_model, processed_data, texts, index = load_data_and_embeddings()
all_field_names = get_all_field_names(CSV_FILE)
TOP_K = len(texts)

# --- App Initialization ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if not st.session_state["messages"]:
    welcome = "\U0001F44B Hello! I can help you with IHRD college information. Ask me anything."
    st.session_state["messages"].append({"role": "assistant", "content": welcome})

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
        st.rerun()
        
    if st.button("📥 Download Chat"):
        if st.session_state["messages"]:
            chat_text = "\n\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state["messages"]])
            st.download_button("Download as TXT", data=chat_text, file_name="chat_history.txt", mime="text/plain")

# --- Chat Display ---
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
    
