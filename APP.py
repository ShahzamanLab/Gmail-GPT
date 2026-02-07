import streamlit as st
from src.Gmail_data_loader import GmailService, GmailLoader
from src.Gmail_data_splitter import DocumentSplitter
from src.Gmail_vectorstore import PineconeVectorStoreManager
from src.Gmail_prompt_loader import PromptLoader
from src.Gmail_data_retriver import VectorStoreRetriever

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv
import os

# ==========================
# Load environment variables
# ==========================
load_dotenv()

# ==========================
# Streamlit App Setup
# ==========================
st.set_page_config(page_title="Gmail RAG Chat", page_icon="🤖")
st.title("🤖 Gmail Assistant Chat")
st.write("Ask questions about your Gmail emails!")

# Initialize session state for conversation
if "history" not in st.session_state:
    st.session_state.history = []

# ==========================
# 1️⃣ Load Emails and Prepare RAG
# ==========================
@st.cache_resource(show_spinner=True)
def initialize_rag():
    # Load Gmail Emails
    gmail_service = GmailService()
    gmail_loader = GmailLoader(gmail_service)
    documents = gmail_loader.load_emails()

    doc_splitter = DocumentSplitter()
    chunks = doc_splitter.split(documents)

    vector_store_manager = PineconeVectorStoreManager(
        index_name="gmail-base-data",
        dimension=384
    )

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    vector_store_manager.add_texts(texts=texts, metadatas=metadatas)

    retriever = VectorStoreRetriever(
        vectorstore=vector_store_manager.vectorstore,
        k=5
    )

    # Load prompt template
    prompt_loader = PromptLoader(base_path="src")
    prompt_text = prompt_loader.load("Gmail_voice_rag_prompt.txt")
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_text
    )

    # Initialize LLM
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.2
    )

    # Build RAG chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    return rag_chain

rag_chain = initialize_rag()

# ==========================
# 2️⃣ Chat Interface
# ==========================
user_input = st.text_input("📝 You:", key="input")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})

    # Invoke RAG chain
    response = rag_chain.invoke(user_input)

    st.session_state.history.append({"role": "assistant", "content": response.content})

# ==========================
# 3️⃣ Display Chat
# ==========================
for chat in st.session_state.history:
    if chat["role"] == "user":
        st.markdown(
            f'<div style="background-color:#DCF8C6;padding:8px;border-radius:10px;margin:5px 0;color:black;"><b>You:</b> {chat["content"]}</div>',
            unsafe_allow_html=True
        )
    else:
        # AI bubble background is light, so use black text
        st.markdown(
            f'<div style="background-color:#F0F0F0;padding:8px;border-radius:10px;margin:5px 0;color:black;"><b>🤖 Areeba:</b> {chat["content"]}</div>',
            unsafe_allow_html=True
        )
