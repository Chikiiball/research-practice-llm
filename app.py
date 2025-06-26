import streamlit as st
from rag_bot import ask
import os
import json

st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("📄 Local PDF Q&A Chatbot")

# Show PDF status
pdf_dir = "pdf-dataset"
pdf_count = len([f for f in os.listdir(pdf_dir) if f.endswith(".pdf")])
st.info(f"📁 Found {pdf_count} PDF(s) in `{pdf_dir}`")

if pdf_count == 0:
    st.warning("⚠️ No PDFs found. Please add PDF files to the `pdf-dataset/` folder and restart the app.")

# Question input
query = st.text_input("Ask a question about your PDFs:")

if query:
    with st.spinner("Thinking..."):
        try:
            label, answer, sources = ask(query)
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()

    st.markdown(f"### {label}")
    st.write(answer)

    if sources:
        st.markdown("### Retrieved Source(s):")
        for i, doc in enumerate(sources):
            st.markdown(f"**Chunk {i+1}:**")
            st.write(doc.page_content)

# Sidebar: Logs
st.sidebar.title("📜 Recent Logs")
log_file = "responseData.json"

def load_logs():
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def save_logs(logs):
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

logs = load_logs()

if logs:
    logs = logs[-5:][::-1]  # latest 5, newest first
    for idx, log in enumerate(logs):
        key = f"log_{idx}"
        with st.sidebar.expander(f"🕒 {log['timestamp']} - {log['response_type']}"):
            st.markdown(f"**Q:** {log['question']}")
            st.markdown(f"**Response time:** {log['response_time_sec']}s")
            st.markdown(f"**A:** {log['response'][:300]}{'...' if len(log['response']) > 300 else ''}")
            if st.button("🗑️ Delete", key=f"delete_{key}"):
                all_logs = load_logs()
                all_logs = [l for l in all_logs if l['timestamp'] != log['timestamp']]
                save_logs(all_logs)
                st.experimental_rerun()
else:
    st.sidebar.info("No logs recorded yet.")
