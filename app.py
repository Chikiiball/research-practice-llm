import streamlit as st
from rag_bot import ask
import os
import json
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="📚 RAG PDF Chatbot", layout="centered", initial_sidebar_state="expanded")

# --- HEADER STYLING ---
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .source-chunk {
        background-color: #f8f9fa;
        padding: 0.75rem;
        border-radius: 8px;
        font-size: 0.95rem;
    }
</style>
<div class="main-title">📚 RAG PDF Chatbot</div>
<div class="subtitle">Ask questions based on your uploaded PDF documents.</div>
""", unsafe_allow_html=True)

# --- PDF CHECK ---
pdf_dir = "pdf-dataset"
pdf_count = len([f for f in os.listdir(pdf_dir) if f.endswith(".pdf")])
st.toast(f"📁 Found {pdf_count} PDF file(s) in `{pdf_dir}/`", icon="📄")

if pdf_count == 0:
    st.warning("⚠️ No PDFs found. Add files to `pdf-dataset/` and restart.")

# --- SESSION STATE INIT ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# --- USER QUESTION ---
query = st.chat_input("Ask a question about your PDFs...")
if query:
    with st.spinner("🔍 Searching for answers..."):
        try:
            label, answer, sources = ask(query)
            response_type = "📄 PDF" if "PDF" in label else "🤖 Model"
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()

        # Save interaction to session state
        st.session_state.messages.append({
            "query": query,
            "response": answer,
            "response_type": response_type,
            "sources": [doc.page_content for doc in sources],
            "timestamp": datetime.now().isoformat()
        })

        # Save to JSON log
        log_file = "responseData.json"
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append({
            "question": query,
            "response": answer,
            "response_type": response_type,
            "response_time_sec": round(datetime.now().timestamp(), 2),
            "timestamp": datetime.now().isoformat()
        })

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)

# --- DISPLAY CHAT ---
for msg in st.session_state.messages:
    with st.chat_message("user"):
        st.markdown(msg["query"])

    with st.chat_message("assistant"):
        st.markdown(f"""
<div style='background-color:#000;padding:1rem;border-radius:10px;margin-bottom:0.5rem;'>
<b>{msg['response_type']} Answer:</b><br><br>
{msg['response']}
</div>
""", unsafe_allow_html=True)

        if msg["sources"]:
            with st.expander("📚 Show Source Chunks"):
                for i, chunk in enumerate(msg["sources"]):
                    st.markdown(f"<div style='background-color:#000' class='source-chunk'><b>Chunk {i+1}:</b><br>{chunk}</div>", unsafe_allow_html=True)

# --- SIDEBAR: INTERACTION LOG ---
st.sidebar.title("📝 Recent Logs")
log_file = "responseData.json"

def load_logs():
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_logs(logs):
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

logs = load_logs()

if logs:
    logs = logs[-5:][::-1]  # Show last 5 entries, latest first
    for idx, log in enumerate(logs):
        with st.sidebar.expander(f"{log['timestamp'].split('T')[0]} | {log['response_type']}"):
            st.markdown(f"**Q:** {log['question']}")
            st.markdown(f"**⏱ Response time:** {log['response_time_sec']}s")
            st.markdown(f"**A:** {log['response'][:150]}{'...' if len(log['response']) > 150 else ''}")
            if st.button("🗑 Delete", key=f"del_{idx}"):
                logs = [l for l in logs if l['timestamp'] != log['timestamp']]
                save_logs(logs)
                st.experimental_rerun()
else:
    st.sidebar.info("No logs recorded yet.")

st.sidebar.markdown("""
---
🧠 Built with LangChain + Ollama + Streamlit
""")
