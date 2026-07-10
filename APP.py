import streamlit as st
from dotenv import load_dotenv
import os
import re
import csv
import io
import time
from datetime import datetime
from typing import List, Dict, Any

from src.Gmail_data_loader import GmailService, GmailLoader
from src.Gmail_data_splitter import DocumentSplitter
from src.Gmail_vectorstore import PineconeVectorStoreManager
from src.Gmail_prompt_loader import PromptLoader
from src.Gmail_Agentic_capibility.email_sender import EmailSender

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

st.set_page_config(
    page_title="SAM · Smart Assistant for Mail",
    page_icon="✉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# SESSION STATE
# ==========================
def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "assistant_name" not in st.session_state:
        st.session_state.assistant_name = "SAM"
    if "selected_tone" not in st.session_state:
        st.session_state.selected_tone = "friendly"
    if "selected_category" not in st.session_state:
        st.session_state.selected_category = "All"
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "total_emails_count" not in st.session_state:
        st.session_state.total_emails_count = 0
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None
    if "confirm_bulk_send" not in st.session_state:
        st.session_state.confirm_bulk_send = False
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False

initialize_session_state()

# ==========================
# LOGIC FUNCTIONS
# ==========================
def build_vector_store(max_results=200, days_back=7, progress_callback=None):
    try:
        if progress_callback:
            progress_callback(0.1, "Connecting to Gmail…")
        gmail_service = GmailService()
        gmail_loader = GmailLoader(gmail_service, max_results=max_results)
        if progress_callback:
            progress_callback(0.3, "Loading emails…")
        documents = gmail_loader.load_emails(days_back=days_back)

        if not documents:
            return None, 0

        if progress_callback:
            progress_callback(0.6, "Splitting documents…")
        doc_splitter = DocumentSplitter()
        chunks = doc_splitter.split(documents)

        if progress_callback:
            progress_callback(0.8, "Building vector index…")
        vector_store_manager = PineconeVectorStoreManager(
            index_name="gmail-assistant-data", dimension=384
        )

        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        vector_store_manager.add_texts(texts=texts, metadatas=metadatas)

        if progress_callback:
            progress_callback(1.0, "Ready!")
        return vector_store_manager, len(documents)
    except Exception as e:
        st.error(f"Couldn't connect to Gmail: {e}")
        return None, 0

@st.cache_resource
def get_email_sender():
    return EmailSender(GmailService())

def process_message(user_input):
    if st.session_state.is_processing:
        st.warning("Please wait for the current response to finish.")
        return

    st.session_state.is_processing = True
    st.session_state.history.append({"role": "user", "content": user_input})

    # Add a placeholder for the assistant's response (will be updated later)
    st.session_state.history.append({"role": "assistant", "content": "⏳ Thinking…"})

    # Build vector store if needed
    if st.session_state.vector_store is None:
        with st.status("Loading your inbox…", expanded=True) as status:
            progress_bar = st.progress(0, text="Starting…")
            def update_progress(val, msg):
                progress_bar.progress(val, text=msg)
                status.write(msg)
            vs, count = build_vector_store(progress_callback=update_progress)
            if vs:
                st.session_state.vector_store = vs
                st.session_state.total_emails_count = count
                status.update(label="Inbox loaded!", state="complete")
            else:
                status.update(label="Failed to load emails", state="error")
                st.session_state.history.pop()  # remove user message
                st.session_state.history.pop()  # remove placeholder
                st.session_state.is_processing = False
                return

    # Prepare the retriever and chain
    retriever = st.session_state.vector_store.as_retriever(
        k=15,
        category=st.session_state.selected_category
    )

    prompt_loader = PromptLoader()
    prompt_text = prompt_loader.build_dynamic_prompt(
        st.session_state.assistant_name,
        st.session_state.selected_tone,
        total_emails=st.session_state.total_emails_count
    )

    prompt = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template=prompt_text
    )
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs) if docs else "No emails found."

    history_str = "\n".join([
        f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
        for m in st.session_state.history[:-2]  # exclude current user and placeholder
    ])

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "chat_history": lambda _: history_str
        }
        | prompt
        | llm
    )

    try:
        response = chain.invoke(user_input)
        response_text = response.content
    except Exception as e:
        response_text = f"Something went wrong while generating a reply: {e}"

    # Handle email sending action
    match = re.search(
        r"\[ACTION:SEND_EMAIL\]\s*TO:\s*(.*?)\s*SUBJECT:\s*(.*?)\s*BODY:\s*(.*?)\s*\[/ACTION\]",
        response_text,
        re.DOTALL | re.IGNORECASE
    )
    if match:
        to, subj, body = match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
        response_text = response_text[:match.start()].strip()
        success, msg = get_email_sender().send_email(to, subj, body)
        response_text += f"\n\n{'✅' if success else '❌'} {msg}"

    # Replace the placeholder with the actual response
    st.session_state.history[-1] = {"role": "assistant", "content": response_text}
    st.session_state.is_processing = False

# ==========================
# SIDEBAR
# ==========================
with st.sidebar:
    st.markdown("### ✉️ SAM")
    st.caption("Smart Assistant for Mail")

    if st.button("✨  New chat", use_container_width=True, type="primary"):
        st.session_state.history = []
        st.session_state.is_processing = False
        st.rerun()

    st.divider()

    st.subheader("Settings")
    assistant_name = st.text_input("Assistant name", value=st.session_state.assistant_name)

    tone_options = {"Friendly": "friendly", "Professional": "professional", "Brutal": "brutal"}
    selected_tone_label = st.selectbox(
        "Tone", options=list(tone_options.keys()),
        index=list(tone_options.values()).index(st.session_state.selected_tone)
    )

    category_options = ["All", "Promotions", "Social", "Updates", "Important", "Personal"]
    selected_category = st.selectbox(
        "Category filter", category_options,
        index=category_options.index(st.session_state.selected_category)
    )

    st.session_state.assistant_name = assistant_name
    st.session_state.selected_tone = tone_options[selected_tone_label]
    st.session_state.selected_category = selected_category

    st.divider()

    with st.expander("📧  Bulk email sender"):
        st.caption("Upload a CSV with **email, subject, body** columns.")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="bulk_csv", label_visibility="collapsed")
        if uploaded_file:
            try:
                reader = csv.DictReader(io.StringIO(uploaded_file.getvalue().decode("utf-8-sig")))
                emails = [{'to': r['email'], 'subject': r['subject'], 'body': r['body']} for r in reader if r.get('email')]
                if emails:
                    st.success(f"✅ {len(emails)} email(s) ready to send")
                    with st.expander("Preview first 3 rows"):
                        for e in emails[:3]:
                            st.write(f"**To:** {e['to']}  \n**Subject:** {e['subject']}")

                    st.session_state.confirm_bulk_send = st.checkbox(
                        f"I confirm I want to send {len(emails)} email(s) now",
                        value=st.session_state.confirm_bulk_send
                    )

                    if st.button("📤 Send All", use_container_width=True, type="primary",
                                 disabled=not st.session_state.confirm_bulk_send):
                        sender = get_email_sender()
                        progress = st.progress(0, text="Sending…")
                        sent, failed = 0, 0
                        for i, e in enumerate(emails):
                            ok, _ = sender.send_email(e['to'], e['subject'], e['body'])
                            sent += 1 if ok else 0
                            failed += 0 if ok else 1
                            progress.progress((i + 1) / len(emails), text=f"Sent {i + 1}/{len(emails)}")
                            time.sleep(1)
                        st.success(f"Done — {sent} sent, {failed} failed.")
                        st.session_state.confirm_bulk_send = False
                        st.rerun()
                else:
                    st.warning("No valid rows found (need an 'email' column).")
            except Exception:
                st.error("That CSV couldn't be read. Check the formatting and try again.")

    st.divider()
    st.caption("Signed in as **Shah Zaman**")

# ==========================
# MAIN CHAT AREA
# ==========================
# Header
st.markdown(f"""
<div>
    <h1>✉️ {st.session_state.assistant_name}</h1>
    <p>Your Gmail inbox, summarized and searchable.</p>
</div>
""", unsafe_allow_html=True)

# Clear chat button (only if there are messages)
if st.session_state.history:
    col_clear, _ = st.columns([1, 6])
    with col_clear:
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.history = []
            st.session_state.is_processing = False
            st.rerun()

st.divider()

# Display chat messages
for idx, msg in enumerate(st.session_state.history):
    avatar = "🧑" if msg["role"] == "user" else "✉️"
    with st.chat_message(msg["role"], avatar=avatar):
        if msg["role"] == "assistant" and msg["content"] == "⏳ Thinking…":
            st.write("⏳ Thinking…")
        else:
            st.write(msg["content"])

# Suggestion cards when no history
if not st.session_state.history:
    st.info(f"How can I help you today? Try one of the ideas below, or ask {st.session_state.assistant_name} anything about your inbox.")
    col1, col2 = st.columns(2)
    suggestions = [
        ("📊", "Summarize emails", "Get a quick overview of your inbox", "Summarize my emails"),
        ("⭐", "Find important", "Highlight urgent and starred emails", "Show important emails"),
        ("✏️", "Draft a reply", "Let me write a professional response", "Draft a reply to the last email"),
        ("📬", "Check unread", "List all emails I haven't read yet", "Check unread emails"),
    ]
    for i, (icon, title, desc, prompt_text) in enumerate(suggestions):
        with col1 if i % 2 == 0 else col2:
            with st.container(border=True):
                st.markdown(f"**{icon} {title}**")
                st.caption(desc)
                if st.button("Ask", key=f"sugg_{i}", use_container_width=True):
                    st.session_state.pending_prompt = prompt_text

# Chat input
user_input = st.chat_input(f"Message {st.session_state.assistant_name}…", key="main_input", disabled=st.session_state.is_processing)

if st.session_state.pending_prompt:
    user_input = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if user_input and not st.session_state.is_processing:
    process_message(user_input)
    st.rerun()

# Footer
st.caption("SAM can make mistakes. Review important emails before sending.")