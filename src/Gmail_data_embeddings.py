from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

class EmbeddingGenerator:
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
        return self.embedding_model