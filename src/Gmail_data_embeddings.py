from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

class EmbeddingGenerator:
    """Generate embeddings using HuggingFace models"""

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs: dict = {"device": "cpu"}
    ):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs=model_kwargs
        )

    def get_embedding_model(self):
        """Return embedding object for LangChain VectorStores"""
        return self.embedding_model

    def embed_texts(self, texts: list[str]):
        """Optional: direct embedding if needed"""
        return self.embedding_model.embed_documents(texts)
