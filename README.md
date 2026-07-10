# 📧 Gmail GPT – Conversational AI Agent for Your Inbox

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG%20Framework-1C3C3C)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic%20Workflows-1C3C3C)](https://www.langchain.com/langgraph)
[![Groq](https://img.shields.io/badge/Groq-LLM%20Inference-F55036)](https://groq.com/)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20DB-000000)](https://www.pinecone.io/)
[![Gmail API](https://img.shields.io/badge/Gmail%20API-Google%20Cloud-EA4335)](https://developers.google.com/gmail/api)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

##  Overview

**Gmail GPT** is a **fully interactive, AI-powered assistant** built with [Streamlit](https://streamlit.io/) that connects directly to your Gmail account and turns your inbox into a conversation. Instead of scrolling, searching, and manually drafting emails, you simply **ask questions, give commands, and let the agent handle the rest** — backed by a Retrieval-Augmented Generation (RAG) pipeline and an agentic email-sending layer.

Whether you're an **individual** trying to keep on top of a busy inbox, a **business** running email outreach at scale, or a **developer** wanting a reusable Gmail-RAG backend, Gmail GPT gives you a transparent, customizable assistant that sits directly on top of your Gmail account.

Built by **Shahzaman**, CEO & Founder of **Phymaco**.

---

##  Features

| Feature | Description |
|---------|-------------|
| **🔗 Gmail Connection** | Authenticates with your Gmail account via the Google Gmail API and loads your mailbox for the assistant to work with. |
| **💬 Chat With Your Inbox** | Ask natural-language questions about your own emails — "What did the client say last week?", "Summarize my unread emails" — powered by a RAG pipeline over your email data. |
| **🪪 Custom Name** | Rename your assistant to anything you like — *Mike*, *Emma*, or any name of your choice. |
| **🎭 Custom Tone / Personality** | Set the assistant's tone to match how you want it to communicate: Friendly, Professional, Sexy, Brutal, or anything else you define. |
| **📤 Send Email via Chat** | Send a single email just by giving the recipient, subject, and body in the chat — no need to open Gmail at all. |
| **📊 Bulk Email via CSV** | Upload a CSV file and send personalized emails to hundreds of recipients in one action. |
| **✅ Format Validation** | Bulk uploads are validated against a strict `email,subject,body` schema — malformed files raise a clear error instead of failing silently. |
| **🎨 Simple, Clean UI** | Streamlit-based interface for chatting, configuring the assistant, and monitoring email sending status. |

---

##  Architecture

The application is built on a **clean separation of concerns** between the RAG/chat pipeline and the agentic email-sending layer:

```
Gmail-GPT/
├── APP.py                                   # Streamlit frontend (UI logic)
├── src/
│   ├── __init__.py
│   ├── Gmail_data_loader.py                 # Loads and authenticates Gmail data via the Gmail API
│   ├── Gmail_data_splitter.py                # Splits/chunks email content for embedding
│   ├── Gmail_data_embeddings.py              # Generates embeddings using HuggingFace models
│   ├── Gmail_data_utils.py                   # Shared helper/utility functions
│   ├── Gmail_vectorstore.py                  # Manages the Pinecone / FAISS vector store
│   ├── Gmail_data_retriver.py                # Retrieves relevant email context for a query
│   ├── Gmail_voice_rag_prompt.txt            # Prompt template for the RAG assistant
│   ├── Gmail_text_rag.py                     # Core text-based RAG chat pipeline
│   ├── Gmail_prompt_loader.py                # Loads and injects prompt/tone configuration
│   └── Gmail_Agentic_capibility/
│       ├── email_sender.py                  # Sends a single email (to, subject, body)
│       └── bulk_email_sender.py             # Validates and sends bulk emails from a CSV
├── Test/
│   └── classes_test.py                       # Unit tests for core classes
├── Documents/
│   └── images/                                # Screenshots used in this README
├── requirements.txt
├── token.json                                 # Auto-generated after first Google OAuth login
├── .env
└── README.md
```

### Backend Modules

| Module | Responsibility |
|--------|-----------------|
| `Gmail_data_loader` | Authenticates with Google and pulls email data from the connected Gmail account. |
| `Gmail_data_splitter` | Breaks down loaded email content into manageable chunks for embedding. |
| `Gmail_data_embeddings` | Converts email chunks into vector embeddings using HuggingFace / Sentence-Transformers. |
| `Gmail_vectorstore` | Stores and manages embeddings using Pinecone (cloud) or FAISS (local). |
| `Gmail_data_retriver` | Retrieves the most relevant email chunks for a given user query. |
| `Gmail_prompt_loader` | Loads the prompt template and injects the assistant's configured name and tone. |
| `Gmail_text_rag` | Orchestrates the end-to-end RAG chat flow — retrieval + LLM response via Groq. |
| `email_sender` | Sends a single email given a recipient, subject, and body. |
| `bulk_email_sender` | Parses and validates a CSV, then sends personalized emails in bulk. |

All modules are designed to be **frontend-agnostic** — they can be reused outside of the Streamlit app, e.g. in a script, notebook, or API.

---

##  Quick Start

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Git Bash (recommended for this project on Windows)
- A Google Cloud project with the Gmail API enabled

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/ShahzamanLab/Gmail-GPT.git
   cd Gmail-GPT
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**

   ```bash
   source venv/Scripts/Activate
   ```

4. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

5. **Launch the app**

   ```bash
   streamlit run APP.py
   ```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 🔑 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
HUGGINGFACE_API_KEY=your_huggingface_api_key
```

### Google Credentials

Download a `credential.json` file from the **Google Cloud Console** (after enabling the Gmail API and creating OAuth 2.0 credentials) and place it in the project root. On first run, Gmail GPT will complete the OAuth flow and generate a `token.json` for future authenticated access.

---

## 📖 Usage Walkthrough

### Step 1: Connect Your Gmail Account

- Launch the app and complete the Google OAuth sign-in.
- Gmail GPT loads and indexes your mailbox in the background.

![Gmail GPT UI](Documents/images/ui%20image.png)

### Step 2: Name & Configure Your Assistant

- Give your assistant a name — *Mike*, *Emma*, or anything you like.
- Choose its tone: Friendly, Professional, Sexy, Brutal, or your own custom style.

### Step 3: Chat With Your Inbox

- Ask questions about your emails in plain language.
- Get summaries, insights, and answers pulled directly from your real inbox data.

### Step 4: Send a Single Email

- Tell the assistant the recipient, subject, and body.
- It sends the email instantly — no need to open Gmail.

### Step 5: Send Bulk Emails via CSV

- Prepare a CSV file with the following required columns:

  ```csv
  email,subject,body
  john@example.com,Welcome Aboard,Hi John, welcome to our platform!
  sara@example.com,Special Offer,Hi Sara, here's 20% off your next order.
  ```

- Upload the file in the app. If the format doesn't match `email,subject,body`, the app raises a clear error instead of failing silently.
- Track delivery status as emails are sent.

![Email Sending Status](Documents/images/email%20sending%20status.png)

> **Note:** Rename image files to remove spaces (e.g. `ui-image.png`, `email-sending-status.png`) for the most reliable rendering across all markdown viewers, and update the links above if your repository path differs.

---

##  Example Use Cases

- **Personal inbox management** — quickly find and summarize important emails without endless scrolling.
- **Customer support follow-ups** — ask the assistant to draft and send replies in your preferred tone.
- **Marketing / outreach campaigns** — upload a CSV and send personalized bulk emails without manual copy-paste.
- **Business reporting** — ask for a summary of the day's/week's email activity instead of digging through hundreds of messages.

---

## 💡 Why We Built This

Sending and managing emails at scale is a real pain point for individuals and businesses:

- **Bulk sending is slow and manual.** Doing it by hand for hundreds of recipients eats up hours of time.
- **Emails land in spam.** Poorly structured bulk emails frequently get flagged. Gmail GPT is built around Google's regulations and keyword guidelines so your emails have a better chance of reaching the inbox, not the spam folder.
- **No visibility into results.** Once you've sent 1,000 emails, tracking replies, insights, and summaries manually is nearly impossible. Because Gmail GPT is conversational, you can simply ask it for a summary or insight instead of digging through your sent folder.

---

## 🔧 Customisation

Gmail GPT's backend is designed to be extensible. You can:

- Swap the vector store between Pinecone (cloud) and FAISS (local) in `Gmail_vectorstore.py`.
- Change the LLM provider or model used in `Gmail_text_rag.py`.
- Add new tone/personality presets in `Gmail_prompt_loader.py`.
- Extend `bulk_email_sender.py` with additional CSV columns (e.g. attachments, CC/BCC) as needed.

---

## 🤝 Contributing

Contributions are welcome! If you have an idea for a new feature, a bug fix, or improved documentation:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/amazing-feature`).
3. Commit your changes (`git commit -m 'Add amazing feature'`).
4. Push to the branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.

Please ensure your code follows the existing style and includes appropriate tests where applicable.

---

## 🌱 Roadmap & Investment Opportunity

Gmail GPT started as a tool to solve a real, everyday problem, and we're now looking to grow it into a full-fledged product with enhanced features and broader capability.

**We're open to partnership and investment collaboration.** If you're interested in exploring this opportunity with us, reach out.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io/) — for making data apps so easy to build.
- [LangChain](https://www.langchain.com/) & [LangGraph](https://www.langchain.com/langgraph) — for the RAG and agentic workflow framework.
- [Groq](https://groq.com/) — for fast LLM inference.
- [Pinecone](https://www.pinecone.io/) & [FAISS](https://github.com/facebookresearch/faiss) — for vector storage and similarity search.
- [Hugging Face](https://huggingface.co/) — for embeddings and model access.
- [Google Gmail API](https://developers.google.com/gmail/api) — for secure inbox access.

---

## 📧 Contact

**Shahzaman** — CEO & Founder, Phymaco

For questions, suggestions, or collaboration, reach out via email:
- gallanizaman@gmail.com
- phymaco.contact@gmail.com

Or visit our website: [phymaco.com](https://phymaco.com/)