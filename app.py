import streamlit as st
from rag_bot import ask
import os

st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("📄 Local PDF Q&A Chatbot")

# Show PDF count
pdf_dir = "pdf-dataset"
pdf_count = len([f for f in os.listdir(pdf_dir) if f.endswith(".pdf")])
st.info(f"📁 Found {pdf_count} PDF(s) in `{pdf_dir}`")

if pdf_count == 0:
    st.warning("⚠️ No PDFs found. Please add PDF files to the `pdf-dataset/` folder and restart the app.")

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
