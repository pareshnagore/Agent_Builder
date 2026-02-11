import uuid
import tiktoken
import nltk
from core.vector_db import VectorDB

# Download sentence tokenizer once
nltk.download("punkt", quiet=True)


class RAGEngine:

    def __init__(self, embed_model="mxbai-embed-large"):
        print("RAGEngine initialized with model:", embed_model)
        self.vectordb = VectorDB(embed_model=embed_model)
        # Use a general tokenizer – works well for most LLMs
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    # -------- TOKEN AWARE CHUNKING ---------

    def token_count(self, text):
        return len(self.tokenizer.encode(text))

    def chunk_text(self, text, max_tokens=400, overlap_tokens=50):
        """
        Token-aware + sentence-aware chunking
        """
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = ""
        current_tokens = 0

        for sent in sentences:
            sent_tokens = self.token_count(sent)
            # If single sentence itself is too large, truncate safely
            if sent_tokens > max_tokens:
                sent = self.tokenizer.decode(
                    self.tokenizer.encode(sent)[:max_tokens]
                )
                sent_tokens = self.token_count(sent)

            # If adding this sentence exceeds limit -> finalize current chunk
            if current_tokens + sent_tokens > max_tokens:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                overlap_text = " ".join(
                    self.tokenizer.decode(
                        self.tokenizer.encode(current_chunk)[-overlap_tokens:]
                    ).split()
                )
                current_chunk = overlap_text + " " + sent
                current_tokens = self.token_count(current_chunk)

            else:
                current_chunk += " " + sent
                current_tokens += sent_tokens

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks


    # -------- INDEXING ---------

    def index_documents(self, texts, source="manual"):

        for text in texts:
            if not text or len(text.strip()) < 20:
                print(f"Skipping invalid document: {source}")
                continue

            existing_sources = self.vectordb.list_sources()
            if source in existing_sources:
                print(f"Source '{source}' already indexed. Skipping.")
                return False    
            chunks = self.chunk_text(text)
            metadatas = [{"source": source} for _ in chunks]
            ids = [str(uuid.uuid4()) for _ in chunks]
            self.vectordb.add(
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
        return True


    # -------- RETRIEVAL ---------

    def retrieve(self, query, top_k=3):

        results = self.vectordb.search(
            query_text=query,
            n_results=top_k
        )
        documents = results.get("documents", [[]])[0]
        context = "\n\n".join(documents)
        return context
