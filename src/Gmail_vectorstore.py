from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_core.retrievers import BaseRetriever  # Updated import
from langchain_core.documents import Document
from typing import List
import os
from dotenv import load_dotenv
load_dotenv()

class PineconeVectorStoreManager:
    def __init__(self, index_name: str, dimension: int = 384):
        self.index_name = index_name

        # Embeddings
        from src.Gmail_data_embeddings import EmbeddingGenerator
        self.embedding = EmbeddingGenerator().get_embedding_model()

        # Pinecone init
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self._get_or_create_index(dimension)

        self.vectorstore = PineconeVectorStore(
            index=self.index,
            embedding=self.embedding
        )

    def _get_or_create_index(self, dimension):
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        return self.pc.Index(self.index_name)

    def add_texts(self, texts, metadatas=None):
        return self.vectorstore.add_texts(texts=texts, metadatas=metadatas)

    def search(self, query, k=3):
        return self.vectorstore.similarity_search(query, k=k)

    # Corrected as_retriever
    def as_retriever(self, k=5):
        class VectorStoreRetriever(BaseRetriever):
            def __init__(self, vectorstore, k: int = 5):
                self.vectorstore = vectorstore
                self.k = k
            
            def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
                return self.vectorstore.similarity_search(query, k=self.k)
            
            async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
                # Optional: Implement async version
                return self.vectorstore.similarity_search(query, k=self.k)
        
        return VectorStoreRetriever(self.vectorstore, k=k)