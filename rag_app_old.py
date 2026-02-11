import streamlit as st
import ollama
import chromadb
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract
import os

from chromadb.api.types import EmbeddingFunction

st.title("Local Personal Document RAG")



# ---------- Embedding Function ----------

class OllamaEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input):
        embeddings = []
        for text in input:
            response = ollama.embeddings(
                model=embed_model,
                prompt=text
            )
            embeddings.append(response["embedding"])
        return embeddings


# ---------- Sidebar Settings ----------

st.sidebar.header("Settings")

embed_model = st.sidebar.selectbox(
    "Embedding Model",
    ["embeddinggemma:latest", "mxbai-embed-large"]
)

chat_model = st.sidebar.selectbox(
    "Choose Model",
    ["gemma2:2b", "llama3.2", "mistral"]
)

category = st.sidebar.selectbox(
    "Document Category",
    ["finance", "identity", "medical", "general"]
)

if st.sidebar.button("Clear Entire Database"):
    chroma_client = chromadb.PersistentClient(path="db")
    chroma_client.reset()
    st.sidebar.success("Database Cleared!")

# ---------- Chroma Setup ----------

embedding_function = OllamaEmbeddingFunction()

chroma_client = chromadb.PersistentClient(path="db")

collection = chroma_client.get_or_create_collection(
    name=f"docs_{category}",
    embedding_function=embedding_function
)

# ---------- Utility Functions ----------

def extract_text_from_pdf(path):
    text = ""

    try:
        reader = PdfReader(path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    except:
        pass

    if not text.strip():
        try:
            images = convert_from_path(path)
            for img in images:
                text += pytesseract.image_to_string(img)
        except:
            pass

    return text


def chunk_text(text, size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def index_pdf(file_path):
    text = extract_text_from_pdf(file_path)

    if not text.strip():
        return "No readable text"

    chunks = chunk_text(text)

    filename = os.path.basename(file_path)

    for i, chunk in enumerate(chunks):
        doc_id = f"{filename}_{i}"

        existing = collection.get(ids=[doc_id])

        if not existing["ids"]:
            collection.add(
                documents=[chunk],
                metadatas=[{
                    "source": filename,
                    "category": category
                }],
                ids=[doc_id]
            )

    return "Indexed"


# ---------- Show Indexed Documents ----------

st.subheader("Already Indexed Documents")

all_data = collection.get()

indexed_files = set()

for meta in all_data.get("metadatas", []):
    indexed_files.add(meta.get("source"))

if indexed_files:
    for f in sorted(indexed_files):
        st.write("📄", f)
else:
    st.write("No documents indexed yet in this category.")

st.markdown("---")

# ---------- Upload Single PDF ----------

st.subheader("Upload Individual PDF")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    save_path = os.path.join("data", uploaded_file.name)

    os.makedirs("data", exist_ok=True)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    status = index_pdf(save_path)

    st.success(f"{uploaded_file.name}: {status}")

st.markdown("---")

# ---------- Folder Indexing Section ----------

st.subheader("Index Entire Folder")

folder_path = st.text_input(
    "Enter full folder path containing PDFs (e.g. /Users/you/Documents/pdfs)"
)

if st.button("Index Folder"):
    if os.path.exists(folder_path):
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]

        if not files:
            st.warning("No PDFs found in folder")
        else:
            progress = st.progress(0)

            for idx, file in enumerate(files):
                full_path = os.path.join(folder_path, file)
                status = index_pdf(full_path)
                st.write(f"{file}: {status}")

                progress.progress((idx + 1) / len(files))

            st.success("Folder indexing completed!")

    else:
        st.error("Invalid folder path")

st.markdown("---")

# ---------- Query Section ----------

st.subheader("Ask Questions")

query = st.text_input("Ask about your documents")

if query:
    results = collection.query(
        query_texts=[query],
        n_results=10
    )

    docs = results["documents"][0]
    metadatas = results["metadatas"][0]

    context = ""
    sources = set()

    for d, m in zip(docs[:5], metadatas[:5]):
        context += d + "\n\n"
        sources.add(m["source"])

    prompt = f"""
    Use ONLY the following context to answer.

    Context:
    {context}

    Question:
    {query}

    If answer is not in context, say "I don't know".
    """

    response = ollama.generate(
        model=chat_model,
        prompt=prompt
    )

    st.write("### Answer")
    st.write(response["response"])

    st.write("### Sources")
    for s in sources:
        st.write("-", s)
