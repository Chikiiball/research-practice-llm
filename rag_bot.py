import os
import time
import json
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

from sentence_transformers import SentenceTransformer, util
import torch

# Load all PDFs from directory
def load_all_pdfs(pdf_dir="pdf-dataset/"):
    documents = []
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_dir, filename))
            documents.extend(loader.load())
    return documents

# Document loading
try:
    documents = load_all_pdfs()
    if not documents:
        raise ValueError("No PDF documents found.")
except Exception as e:
    print(f"[ERROR] Failed to load PDFs: {e}")
    documents = []

# Prepare models and vectorstore
if documents:
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(texts, embedding)

    llm = OllamaLLM(model="deepseek-r1")

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True
    )

    semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def is_context_relevant(question, docs, threshold=0.6):
    question_emb = semantic_model.encode(question, convert_to_tensor=True)
    max_score = 0

    for doc in docs:
        context_emb = semantic_model.encode(doc.page_content, convert_to_tensor=True)
        sim = util.pytorch_cos_sim(question_emb, context_emb).item()
        max_score = max(max_score, sim)

    print(f"[DEBUG] Max similarity score: {max_score:.2f}")
    return max_score > threshold

def ask(question: str):
    if not documents:
        raise RuntimeError("No documents loaded. Cannot answer questions.")

    start_time = time.time()
    print(f"[DEBUG] Question: {question}")
    result = qa({"query": question})
    answer = result.get('result', '')
    sources = result.get('source_documents', [])
    response_type = "From PDF"

    print(f"[DEBUG] Retrieved answer: {answer}")
    print(f"[DEBUG] Number of source docs: {len(sources)}")

    if not sources or not is_context_relevant(question, sources):
        print("[DEBUG] Falling back to LLM...")
        try:
            answer = llm.invoke(f"Answer this question directly: {question}")
            response_type = "From Model"
        except Exception as e:
            answer = f"Error getting answer from model: {e}"
            response_type = "Error"

    end_time = time.time()

    # Save log
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "response": answer,
        "response_type": response_type,
        "response_time_sec": round(end_time - start_time, 2)
    }
    save_log(log_entry)

    return f"Answer ({response_type}):", answer, sources

def save_log(entry, path="responseData.json"):
    try:
        logs = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                logs = json.load(f)
        logs.append(entry)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Failed to save log: {e}")
