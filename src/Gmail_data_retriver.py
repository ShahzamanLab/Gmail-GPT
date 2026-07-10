from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List, Optional, Dict, Any
from pydantic import PrivateAttr

class VectorStoreRetriever(BaseRetriever):
    _vectorstore: Any = PrivateAttr()
    _k: int = PrivateAttr()
    _filter: Optional[Dict] = PrivateAttr(default=None)

    def __init__(self, vectorstore, k: int = 5, filter_dict: Optional[Dict] = None):
        super().__init__()
        self._vectorstore = vectorstore
        self._k = k
        self._filter = filter_dict

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager=None
    ) -> List[Document]:
        if self._filter:
            return self._vectorstore.similarity_search(query, k=self._k, filter=self._filter)
        return self._vectorstore.similarity_search(query, k=self._k)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager=None
    ) -> List[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)