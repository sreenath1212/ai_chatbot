import csv
import re
import multiprocessing
from multiprocessing import Pool
import streamlit as st
import faiss
from sentence\_transformers import SentenceTransformer
import numpy as np
import requests
import os
import time
import json
import google.generativeai as genai

# --- MUST BE FIRST: Streamlit page config ---

st.set\_page\_config(
page\_title="🎓 College Info Assistant",
page\_icon="🤖",
layout="wide",
initial\_sidebar\_state="collapsed"
)

# Inject responsive viewport meta tag

st.markdown(""" <meta name="viewport" content="width=device-width, initial-scale=1.0">
""", unsafe\_allow\_html=True)

# --- Centered Title and Subtitle ---

st.markdown(""" <style>
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
} </style>

```
<div class="centered-title">🎓 College Info Assistant</div>
<div class="centered-subtitle">
    An Intelligent Chatbot for College Search Powered by FAISS, Sentence Transformers, and Gemini Flash
    <br>ALL MESSAGES ARE CURRENTLY MONITORED BY ADMIN FOR IMPROVING CHATBOT PERFORMANCE
</div>
<hr>
```

""", unsafe\_allow\_html=True)

# --- Configuration ---

CSV\_FILE = 'standardized\_finatdata.csv'
TXT\_FILE = 'institution\_descriptions.txt'

# --- Gemini Model Setup ---

genai.configure(api\_key=st.secrets\["GEMINI\_API\_KEY"])
llm\_model = genai.GenerativeModel('gemini-2.0-flash')

# --- Utility Functions ---

def clean\_field\_name(field\_name):
field\_name = field\_name.replace('\_', ' ').replace('\n', ' ').strip().capitalize()
field\_name = re.sub(' +', ' ', field\_name)
return field\_name

def process\_row(row):
description = ""
institution\_name = row\.get('name\_of\_the\_institution\_full\_name', '').strip()
if institution\_name:
description += f"{institution\_name}."
else:
description += "Institution Name: Not Available."

```
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
```

def generate\_metadata\_from\_csv(csv\_filepath, output\_txt\_path, num\_workers=None):
if os.path.exists(output\_txt\_path):
return
with open(csv\_filepath, 'r', encoding='utf-8') as csvfile:
reader = list(csv.DictReader(csvfile))
with Pool(processes=num\_workers or multiprocessing.cpu\_count()) as pool:
paragraphs = pool.map(process\_row, reader)
with open(output\_txt\_path, 'w', encoding='utf-8') as outfile:
for paragraph in paragraphs:
outfile.write(paragraph + '\n' + '-' \* 40 + '\n')

@st.cache\_resource
def load\_data\_and\_embeddings():
with open(TXT\_FILE, 'r', encoding='utf-8') as file:
texts = file.read().split('----------------------------------------')
texts = \[text.strip() for text in texts if text.strip()]
embedding\_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedding\_model.encode(texts, show\_progress\_bar=True)
index = faiss.IndexFlatL2(embeddings.shape\[1])
index.add(np.array(embeddings))
return embedding\_model, texts, index

def retrieve\_relevant\_context(query, top\_k):
query\_emb = embedding\_model.encode(\[query])
distances, indices = index.search(np.array(query\_emb), top\_k)
context = "\n\n".join(\[texts\[i] for i in indices\[0]])
return context

def ask\_gemini(context, question):
prompt = f"""
You are a helpful, intelligent assistant that provides concise, accurate answers about colleges.

Only include relevant, available information. Omit fields that are missing or marked 'Nil'. Use clear, professional language.

### CONTEXT:

{context}

### QUESTION:

{question}

### INSTRUCTIONS:

* Answer only what the user asked.
* Omit unrelated or unavailable details.
* Expand abbreviations (e.g., BSc → Bachelor of Science).
* If asked about a specific field , list all the names in it.
* Never say "data not available".
* Be precise, use complete sentences.

"""
print("\n\n--- FINAL PROMPT TO GEMINI ---\n")
print(prompt)
print("\n--- END PROMPT ---\n")
try:
response = llm\_model.generate\_content(prompt)
return response.text
except Exception as e:
return f"❌ Gemini error: {e}"

# --- Memory Persistence ---

MEMORY\_FILE = "chat\_memory.json"

def save\_memory():
with open(MEMORY\_FILE, "w", encoding="utf-8") as f:
json.dump(st.session\_state\["messages"], f)

def load\_memory():
if os.path.exists(MEMORY\_FILE):
with open(MEMORY\_FILE, "r", encoding="utf-8") as f:
st.session\_state\["messages"] = json.load(f)

# --- Main App Logic ---

generate\_metadata\_from\_csv(CSV\_FILE, TXT\_FILE)
embedding\_model, texts, index = load\_data\_and\_embeddings()
TOP\_K = len(texts)

if "messages" not in st.session\_state:
st.session\_state\["messages"] = \[]
load\_memory()

if not st.session\_state\["messages"]:
welcome\_message = "👋 Hello! How can I help you today? I can assist you with any college information you need."
st.session\_state\["messages"].append({"role": "assistant", "content": welcome\_message})
save\_memory()

# --- Sidebar: Chat History ---

with st.sidebar:
st.header("🕑 Chat History")
if st.session\_state\["messages"]:
for msg in st.session\_state\["messages"]:
st.markdown(f"**{msg\['role'].capitalize()}**: {msg\['content']\[:30]}...")
else:
st.markdown("*No chats yet.*")
if st.button("🧹 Clear Chat"):
st.session\_state\["messages"] = \[]
save\_memory()
st.rerun()
if st.button("📥 Download Chat"):
if st.session\_state\["messages"]:
chat\_text = "\n\n".join(\[f"{m\['role'].capitalize()}: {m\['content']}" for m in st.session\_state\["messages"]])
st.download\_button("Download as TXT", data=chat\_text, file\_name="chat\_history.txt", mime="text/plain")

# --- Chat Interface ---

for msg in st.session\_state\["messages"]:
with st.chat\_message(msg\["role"]):
st.markdown(f"<div class='chat-bubble'>{msg\['content']}</div>", unsafe\_allow\_html=True)

user\_query = st.chat\_input("Type your question here...")

if user\_query:
st.session\_state\["messages"].append({"role": "user", "content": user\_query})
with st.chat\_message("user"):
st.markdown(f"<div class='chat-bubble'>{user\_query}</div>", unsafe\_allow\_html=True)
with st.spinner("Thinking..."):
context = retrieve\_relevant\_context(user\_query, TOP\_K)
raw\_answer = ask\_gemini(context, user\_query)
final\_answer = ""
with st.chat\_message("assistant"):
answer\_placeholder = st.empty()
for i in range(len(raw\_answer)):
final\_answer = raw\_answer\[:i+1]
answer\_placeholder.markdown(f"<div class='chat-bubble'>{final\_answer}</div>", unsafe\_allow\_html=True)
time.sleep(0.01)
st.session\_state\["messages"].append({"role": "assistant", "content": raw\_answer})
save\_memory()
