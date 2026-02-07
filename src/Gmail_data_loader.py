import os
import base64
from typing import List
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
EMAIL_FETCH_LIMIT = 50            # Number of emails to fetch daily

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
        # Load existing token if exists
        if os.path.exists(TOKEN_FILE):
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        
        # If token is invalid or missing, authenticate
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())  # Refresh token if expired
            else:
                # Run OAuth flow for desktop app
                flow = InstalledAppFlow.from_client_secrets_file(CREDS_FILE, SCOPES)
                creds = flow.run_local_server(port=8080)  # Opens browser automatically
            
            # Save the credentials for next time
            with open(TOKEN_FILE, "w") as f:
                f.write(creds.to_json())
        
        # Build Gmail API service
        service = build("gmail", "v1", credentials=creds, cache_discovery=False)
        return service

# ----------------- GMAIL LOADER CLASS -----------------
class GmailLoader:
    """
    Loads emails from Gmail and converts them into LangChain Documents.
    """
    def __init__(self, service: GmailService, limit: int = EMAIL_FETCH_LIMIT):
        self.service = service.service
        self.limit = limit

    def load_emails(self) -> List[Document]:
        """
        Fetch emails from Gmail and return as a list of LangChain Documents.
        """
        # List messages (latest first)
        results = self.service.users().messages().list(
            userId="me",
            maxResults=self.limit,
            q=""  # Optional: filter query, e.g., "is:unread" or "after:2026/01/01"
        ).execute()
        
        messages = results.get("messages", [])
        documents = []

        # Parse each message
        for msg in messages:
            text, metadata = self._parse_message(msg["id"])
            documents.append(Document(page_content=text, metadata=metadata))

        return documents

    def _parse_message(self, msg_id: str):
        """
        Parse a single Gmail message by ID and extract subject, body, and metadata.
        """
        msg = self.service.users().messages().get(userId="me", id=msg_id, format="full").execute()
        payload = msg["payload"]
        headers = {h["name"].lower(): h["value"] for h in payload.get("headers", [])}

        subject = headers.get("subject", "(No Subject)")
        sender = headers.get("from", "(Unknown Sender)")
        date = headers.get("date", "")
        snippet = msg.get("snippet", "")

        # Extract the body (plain text or HTML)
        body = self._get_body(payload)
        
        # Combine for LangChain Document
        text = f"Subject: {subject}\nFrom: {sender}\nDate: {date}\n\n{body}"
        metadata = {
            "msg_id": msg_id,
            "threadId": msg.get("threadId", ""),
            "subject": subject,
            "sender": sender,
            "date": date,
            "snippet": snippet
        }

        return text, metadata

    def _get_body(self, payload):
        """
        Extract plain-text email body from Gmail payload. Fallback to HTML stripping.
        """
        parts = payload.get("parts", [])
        body = ""

        if parts:
            # First try text/plain
            for p in parts:
                if p["mimeType"] == "text/plain":
                    data = p["body"].get("data")
                    if data:
                        body = base64.urlsafe_b64decode(data).decode("utf-8", "ignore")
                        return body
            # Fallback to HTML
            for p in parts:
                if p["mimeType"] == "text/html":
                    data = p["body"].get("data")
                    if data:
                        html = base64.urlsafe_b64decode(data).decode("utf-8", "ignore")
                        body = BeautifulSoup(html, "html.parser").get_text()
                        return body
        else:
            # If no parts, check payload body directly
            data = payload.get("body", {}).get("data")
            if data:
                body = base64.urlsafe_b64decode(data).decode("utf-8", "ignore")

        return body

# ----------------- MAIN -----------------
if __name__ == "__main__":
    # Step 1: Authenticate Gmail and create service
    gmail_service = GmailService()

    # Step 2: Load emails
    loader = GmailLoader(gmail_service)
    documents = loader.load_emails()

    # Step 3: Print a preview of loaded emails
    for i, doc in enumerate(documents):
        print(f"--- Email {i+1} ---")
        print(f"Subject: {doc.metadata['subject']}")
        print(doc.page_content[:300], "...\n")  # first 300 chars
        
