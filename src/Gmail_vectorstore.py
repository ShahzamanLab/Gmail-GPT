import os
from typing import List, Optional, Any
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field

from src.Gmail_data_embeddings import EmbeddingGenerator

load_dotenv()


class CategoryFilterRetriever(BaseRetriever):
    vectorstore: Any = Field(description="The vectorstore to search")
    k: int = Field(default=15, description="Number of documents to return")
    category_filter: Optional[str] = Field(default=None, description="Optional category filter")

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        kwargs = {"k": self.k}
        
        if self.category_filter and self.category_filter != "All":
            kwargs["filter"] = {"category": self.category_filter}
            
        return self.vectorstore.similarity_search(query, **kwargs)

    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)


class PineconeVectorStoreManager:
    def __init__(self, index_name: str, dimension: int = 384):
        self.index_name = index_name
        self.embedding = EmbeddingGenerator().get_embedding_model()
        
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

    def clear_index(self):
        """Clear all data from the index"""
        try:
            self.index.delete(delete_all=True)
            print("✅ Cleared Pinecone index")
        except Exception as e:
            print(f"Warning: Could not clear index: {e}")

    def add_texts(self, texts, metadatas=None):
        """Add texts WITHOUT clearing existing data (incremental)"""
        if not texts:
            print("️ No texts to add")
            return []
        
        print(f" Adding {len(texts)} new emails to vector store...")
        return self.vectorstore.add_texts(texts=texts, metadatas=metadatas)

    def as_retriever(self, k=15, category=None):
        """
        Create retriever with fixed k=15 (or custom)
        
        Args:
            k: Number of emails to retrieve (default: 15)
            category: Filter by category
        """
        return CategoryFilterRetriever(
            vectorstore=self.vectorstore, 
            k=k, 
            category_filter=category
        )