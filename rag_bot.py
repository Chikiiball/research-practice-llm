import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

from sentence_transformers import SentenceTransformer, util
import torch

# Load all PDFs from the directory
def load_all_pdfs(pdf_dir="pdf-dataset/"):
    documents = []
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            documents.extend(docs)
    return documents

# Load documents safely
try:
    documents = load_all_pdfs()
    if not documents:
        raise ValueError("No PDF documents found.")
except Exception as e:
    print(f"[ERROR] Failed to load PDFs: {e}")
    documents = []

# Process if documents are loaded
if documents:
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Embedding and Vector Store
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(texts, embedding)

    # LLM
    llm = OllamaLLM(model="deepseek-r1")

    # Retrieval QA Chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True
    )

    # Semantic Similarity Model
    semantic_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def is_context_relevant(question, docs, threshold=0.4):
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

    print(f"[DEBUG] Question: {question}")
    result = qa({"query": question})
    answer = result.get('result', '')
    sources = result.get('source_documents', [])

    print(f"[DEBUG] Retrieved answer: {answer}")
    print(f"[DEBUG] Number of source docs: {len(sources)}")

    if sources and is_context_relevant(question, sources):
        return "Answer (From PDF):", answer, sources
    else:
        try:
            model_answer = llm.invoke(f"Answer this question directly: {question}")
            print(f"[DEBUG] Model direct answer: {model_answer}")
        except Exception as e:
            model_answer = f"Error getting answer from model: {e}"
            print(f"[ERROR] {model_answer}")
        return "Answer (From model):", model_answer, []
