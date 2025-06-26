import streamlit as st
from rag_bot import ask
import os
import json

st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("📄 Local PDF Q&A Chatbot")

# Directory setup
pdf_dir = "pdf-dataset"
response_log_file = "responseData.json"

# PDF count display
pdf_count = len([f for f in os.listdir(pdf_dir) if f.endswith(".pdf")])
st.info(f"📁 Found {pdf_count} PDF(s) in `{pdf_dir}`")

if pdf_count == 0:
    st.warning("⚠️ No PDFs found. Please add PDF files to the `pdf-dataset/` folder and restart the app.")
    st.stop()

# Ask a question
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

# Sidebar: View previous logs
st.sidebar.title("📜 Recent Logs")

if os.path.exists(response_log_file):
    try:
        with open(response_log_file, "r", encoding="utf-8") as f:
            logs = json.load(f)
        if logs:
            logs = logs[-5:][::-1]  # show last 5 logs, newest first
            for i, log in enumerate(logs):
                with st.sidebar.expander(f"🕒 {log['timestamp']} - {log['response_type']}"):
                    st.markdown(f"**Q:** {log['question']}")
                    st.markdown(f"**Response time:** {log['response_time_sec']}s")
                    st.markdown(f"**A:** {log['response'][:300]}{'...' if len(log['response']) > 300 else ''}")
        else:
            st.sidebar.info("No logs recorded yet.")
    except Exception as e:
        st.sidebar.error(f"Failed to load logs: {e}")
else:
    st.sidebar.info("No response log file found.")
