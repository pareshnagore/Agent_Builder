import ollama


class EmbeddingModel:

    def __init__(self, model="mxbai-embed-large"):
        self.model = model

    def list_models(self):
        return [
            "mxbai-embed-large",
            "embeddinggemma:latest"
        ]

    def embed(self, text):
        response = ollama.embeddings(
            model=self.model,
            prompt=text
        )

        return response["embedding"]
