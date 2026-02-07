from src.Gmail_data_loader import GmailLoader, GmailService
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS  # Fixed import
from langchain_classic.chains import RetrievalQA  # Fixed import
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_KEY")


gmail_service = GmailService()
loader = GmailLoader(gmail_service)


print("Loading emails...")
documents = loader.load_emails()  
print(f"Loaded {len(documents)} emails.")

# 3️⃣ Split emails into manageable chunks for RAG
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,         # ~1000 characters per chunk
    chunk_overlap=300,       # 300 char overlap for context
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
splitted_docs = splitter.split_documents(documents)
print(f"Split into {len(splitted_docs)} chunks.")

# 4️⃣ Generate embeddings for each chunk
embedding_model = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}  # Change to "cuda" if GPU available
)
vector_store = FAISS.from_documents(splitted_docs, embedding_model)
print("Vector store created.")

# 5️⃣ Create LLM for answering questions
llm = ChatGroq(
    model="openai/gpt-oss-20b",  # Example Groq model (change to your preferred one)
)

# 6️⃣ Set up retrieval-based QA chain
# FIXED: Use {question} instead of {query} because RetrievalQA passes 'question' to the prompt
prompt_template = """
You are an assistant summarizing Gmail emails.

Context:
{context}

Question:
{question}

Answer in detail, clearly, and concisely.
"""

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]  # FIXED: Must include both variables
        )
    }
)

# 7️⃣ Function to query Gmail
def query_emails(user_query):
    # FIXED: Use invoke() instead of run() and pass a dictionary with "query" key
    result = qa_chain.invoke({"query": user_query})
    
    # Handle the result dictionary
    print("Answer:", result["result"])
    
    # Optional: Show which emails were used
    if "source_documents" in result:
        print("\nReferenced emails:")
        for i, doc in enumerate(result["source_documents"][:3], 1):
            subject = doc.metadata.get('subject', 'No subject')
            sender = doc.metadata.get('from', 'Unknown sender')
            print(f"{i}. From: {sender} | Subject: {subject}")
            print(f"   Preview: {doc.page_content[:100]}...")
    
    return result["result"]

# Example usage
if __name__ == "__main__":
    try:
        user_input = input("Enter your query about your emails: ")
        query_emails(user_input)
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")