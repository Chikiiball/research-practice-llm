import streamlit as st
from rag_bot import ask
import os
import json
from datetime import datetime

# Load logs
LOG_FILE = "responseData.json"

def load_logs():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    return []

def save_logs(data):
    with open(LOG_FILE, "w") as f:
        json.dump(data, f, indent=2)

def add_log(entry):
    logs = load_logs()
    logs.append(entry)
    save_logs(logs)

# App UI setup
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.markdown("<h1 style='text-align: center;'>📄 RAG PDF Chatbot</h1>", unsafe_allow_html=True)

# Show PDF count
pdf_dir = "pdf-dataset"
pdf_count = len([f for f in os.listdir(pdf_dir) if f.endswith(".pdf")])
st.info(f"📁 Found {pdf_count} PDF(s) in `{pdf_dir}`")

if pdf_count == 0:
    st.warning("⚠️ No PDFs found. Please add PDF files to the `pdf-dataset/` folder and restart the app.")

query = st.text_input("Ask a question about your PDFs:")

if query:
    with st.spinner("Thinking..."):
        start_time = datetime.now()
        try:
            label, answer, sources = ask(query)
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()
        end_time = datetime.now()

        response_time = (end_time - start_time).total_seconds()
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": query,
            "response": answer,
            "response_time_sec": round(response_time, 2),
            "response_type": label.replace("Answer (", "").replace("):", "")
        }
        add_log(log_entry)

    # Chat message style
    with st.chat_message("user"):
        st.markdown(f"**You:** {query}")

    with st.chat_message("assistant"):
        st.markdown(f"""
        <div style='background-color:#eef1f5;color:#000;padding:1rem;border-radius:10px;margin-bottom:0.5rem;'>
        <b>{label}:</b><br><br>
        {answer}
        </div>
        """, unsafe_allow_html=True)

    # Only show source chunks if answer was from PDF
    if label == "Answer (From PDF)" and sources:
        st.markdown("### 📚 Source(s) Retrieved:")
        for i, doc in enumerate(sources):
            with st.expander(f"Chunk {i+1}"):
                st.write(doc.page_content)


# --- DISPLAY RECENT LOGS PREVIEW ---
st.markdown("---")
with st.expander("🕑 Recent Interactions (Logs)", expanded=False):
    logs = load_logs()
    if logs:
        logs = logs[-10:][::-1]  # Last 5 logs
        for i, log in enumerate(logs):
            col1, col2 = st.columns([0.9, 0.1])
            with col1:
                st.markdown(f"**🗨️ Q{i+1}:** {log['question']}")
                preview = log['response'][:150] + ("..." if len(log['response']) > 150 else "")
                with st.expander("Show full answer"):
                    st.write(log['response'])
                st.caption(f"📅 {log['timestamp'].split('T')[0]} | ⏱ {log['response_time_sec']}s | {log['response_type']}")
            with col2:
                if st.button("🗑", key=f"del_preview_{i}"):
                    logs = [l for l in logs if l['timestamp'] != log['timestamp']]
                    save_logs(logs)
                    st.experimental_rerun()
    else:
        st.info("No logs recorded yet.")

# --- Sidebar footer ---
st.sidebar.markdown("## ⚙️ Settings & Info")
st.sidebar.markdown("""
- 🧠 Powered by **LangChain**, **Ollama**, **HuggingFace**, **Streamlit**
- 📁 Docs analyzed from: `pdf-dataset/`
- 💾 Log file: `responseData.json`
""")
