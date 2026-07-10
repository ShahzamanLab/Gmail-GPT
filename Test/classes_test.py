import os
import base64
from typing import List, Dict, Optional
from datetime import datetime
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain_core.documents import Document
from bs4 import BeautifulSoup

# ----------------- CONFIG -----------------
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
TOKEN_FILE = "token.json"         # Stores the access & refresh token
CREDS_FILE = "credentials.json"   # Your OAuth credentials
EMAIL_FETCH_LIMIT = 50            # Number of emails to fetch daily (total across all)

# Gmail category labels mapping
CATEGORY_LABELS = {
    "CATEGORY_PERSONAL": "Primary",
    "CATEGORY_SOCIAL": "Social",
    "CATEGORY_PROMOTIONS": "Promotions",
    "CATEGORY_UPDATES": "Updates",
    "CATEGORY_FORUMS": "Forums",
}

# ----------------- GMAIL SERVICE CLASS -----------------
class GmailService:
    """
    Handles Gmail authentication and API service creation.
    """
    def __init__(self):
        self.service = self._authenticate_gmail()

    def _authenticate_gmail(self):
        """
        Authenticate user with Gmail API and return the service object.
        Creates token.json if not exists or refreshes it if expired.
        """
        creds = None
        if os.path.exists(TOKEN_FILE):
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(CREDS_FILE, SCOPES)
                creds = flow.run_local_server(port=8080)
            
            with open(TOKEN_FILE, "w") as f:
                f.write(creds.to_json())
        
        service = build("gmail", "v1", credentials=creds, cache_discovery=False)
        return service

# ----------------- GMAIL LOADER CLASS (with categorization) -----------------
class GmailLoader:
    """
    Loads emails from Gmail, categorizes them, and converts into LangChain Documents.
    """
    def __init__(self, service: GmailService, limit: int = EMAIL_FETCH_LIMIT):
        self.service = service.service
        self.limit = limit

    def load_emails(self) -> List[Document]:
        """
        Fetch emails from Gmail, assign category, and return as Documents.
        """
        # List messages (latest first) – we request labelIds to get categories
        results = self.service.users().messages().list(
            userId="me",
            maxResults=self.limit,
            q=""  # You can add a filter, e.g., "after:2026/01/01"
        ).execute()
        
        messages = results.get("messages", [])
        documents = []

        for msg in messages:
            text, metadata = self._parse_message(msg["id"])
            # Add category to metadata
            metadata["category"] = self._get_category(msg)  # we need full message to get labels
            documents.append(Document(page_content=text, metadata=metadata))

        return documents

    def _parse_message(self, msg_id: str):
        """
        Parse a single Gmail message by ID and extract subject, body, and metadata.
        The message is fetched with full format including labels.
        """
        msg = self.service.users().messages().get(
            userId="me", id=msg_id, format="full"
        ).execute()
        
        payload = msg["payload"]
        headers = {h["name"].lower(): h["value"] for h in payload.get("headers", [])}

        subject = headers.get("subject", "(No Subject)")
        sender = headers.get("from", "(Unknown Sender)")
        date = headers.get("date", "")
        snippet = msg.get("snippet", "")

        # Extract body (plain text or HTML)
        body = self._get_body(payload)
        
        text = f"Subject: {subject}\nFrom: {sender}\nDate: {date}\n\n{body}"
        metadata = {
            "msg_id": msg_id,
            "threadId": msg.get("threadId", ""),
            "subject": subject,
            "sender": sender,
            "date": date,
            "snippet": snippet,
            "label_ids": msg.get("labelIds", []),  # store raw labels for debugging
        }
        return text, metadata

    def _get_body(self, payload):
        """
        Extract plain-text email body from Gmail payload. Fallback to HTML stripping.
        """
        parts = payload.get("parts", [])
        body = ""

        if parts:
            for p in parts:
                if p["mimeType"] == "text/plain":
                    data = p["body"].get("data")
                    if data:
                        body = base64.urlsafe_b64decode(data).decode("utf-8", "ignore")
                        return body
            for p in parts:
                if p["mimeType"] == "text/html":
                    data = p["body"].get("data")
                    if data:
                        html = base64.urlsafe_b64decode(data).decode("utf-8", "ignore")
                        body = BeautifulSoup(html, "html.parser").get_text()
                        return body
        else:
            data = payload.get("body", {}).get("data")
            if data:
                body = base64.urlsafe_b64decode(data).decode("utf-8", "ignore")
        return body

    def _get_category(self, msg: Dict) -> str:
        """
        Determine the Gmail category from the message's labelIds.
        Returns one of: 'Primary', 'Social', 'Promotions', 'Updates', 'Forums', or 'Other'.
        """
        label_ids = msg.get("labelIds", [])
        for label, category in CATEGORY_LABELS.items():
            if label in label_ids:
                return category
        return "Other"

    # Optional: method to load emails by specific category
    def load_emails_by_category(self, category: str) -> List[Document]:
        """
        Load emails that belong to a specific category (e.g., 'Primary').
        This re‑fetches all emails and filters them, or you can use a Gmail query.
        """
        all_docs = self.load_emails()
        return [doc for doc in all_docs if doc.metadata.get("category") == category]

# ----------------- MAIN (example usage) -----------------
if __name__ == "__main__":
    # Step 1: Authenticate
    gmail_service = GmailService()

    # Step 2: Load all emails (categorized)
    loader = GmailLoader(gmail_service)
    documents = loader.load_emails()

    # Step 3: Print preview grouped by category
    categories = {}
    for doc in documents:
        cat = doc.metadata.get("category", "Other")
        categories.setdefault(cat, []).append(doc)

    for cat, docs in categories.items():
        print(f"\n=== Category: {cat} ({len(docs)} emails) ===")
        for i, doc in enumerate(docs[:3]):  # show first 3 per category
            print(f"  {i+1}. Subject: {doc.metadata['subject']}")
            print(f"     Snippet: {doc.metadata['snippet'][:80]}...")
        if len(docs) > 3:
            print(f"  ... and {len(docs)-3} more")