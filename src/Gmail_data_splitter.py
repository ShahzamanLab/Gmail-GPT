from pydoc import text
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.Gmail_data_loader import GmailLoader, GmailService
import os
from typing import List,Optional
Gmail_service = GmailService()
Gmail_loader = GmailLoader(Gmail_service)
Gmail_loader.load_emails()


class DocumentSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 300,
        separators: Optional[list] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=self.separators
        )
    
    def split(self, documents):
        splitted_docs = self.splitter.split_documents(documents)
        print(f"Split into {len(splitted_docs)} chunks.")
        return splitted_docs


        



    



        
        