from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List
from pydantic import PrivateAttr
from typing import Any


class VectorStoreRetriever(BaseRetriever):
    _vectorstore: Any = PrivateAttr()
    _k: int = PrivateAttr()

    def __init__(self, vectorstore, k: int = 5):
        super().__init__()
        self._vectorstore = vectorstore
        self._k = k

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager=None
    ) -> List[Document]:
        return self._vectorstore.similarity_search(query, k=self._k)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager=None
    ) -> List[Document]:
        return self._vectorstore.similarity_search(query, k=self._k)
