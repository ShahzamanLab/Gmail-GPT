from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Optional

class DocumentSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[list] = None
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=separators or ["\n\n", "\n", " ", ""]
        )
    
    def split(self, documents):
        splitted_docs = self.splitter.split_documents(documents)
        # Ensure category metadata is strictly preserved in every chunk
        for doc in splitted_docs:
            if 'category' not in doc.metadata:
                doc.metadata['category'] = 'Personal'
                
        print(f"✅ Split into {len(splitted_docs)} chunks (metadata preserved).")
        return splitted_docs