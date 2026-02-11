import chromadb
from chromadb.api.types import EmbeddingFunction
import ollama


class OllamaEmbeddingFunction(EmbeddingFunction):

    def __init__(self, model="mxbai-embed-large"):
        self.model = model

    def __call__(self, input):
        embeddings = []
        for text in input:
            response = ollama.embeddings(
                model=self.model,
                prompt=text
            )
            embeddings.append(response["embedding"])
        return embeddings


class VectorDB:

    def __init__(self, persist_dir="data/vector_store", embed_model="mxbai-embed-large"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        embedding_function = OllamaEmbeddingFunction(embed_model)
        collection_name = f"docs_{embed_model.replace(':', '_')}"
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )

    def add(self, documents, metadatas, ids):
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def search(self, query_text, n_results=3):
        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )

    def list_sources(self):
        data = self.collection.get()
        sources = set()
        for m in data.get("metadatas", []):
            if m and "source" in m:
                sources.add(m["source"])
        return sorted(list(sources))

    def reset(self):
        self.client.reset()
