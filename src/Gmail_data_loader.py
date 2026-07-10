import os
import base64
import json
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain_core.documents import Document

SCOPES = ['https://mail.google.com/']

class GmailService:
    def __init__(self, credentials_path: str = "credentials.json", token_path: str = "token.json"):
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.service = self._authenticate()

    def _authenticate(self):
        creds = None
        if os.path.exists(self.token_path):
            creds = Credentials.from_authorized_user_file(self.token_path, SCOPES)
        
        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
            with open(self.token_path, 'w') as token:
                token.write(creds.to_json())
        
        return build('gmail', 'v1', credentials=creds)


class GmailLoader:
    def __init__(self, gmail_service: GmailService, max_results: int = 200):
        self.service = gmail_service.service
        self.max_results = max_results
        self.sync_file = "email_sync_state.json"

    def _get_last_sync_date(self) -> Optional[str]:
        """Get the date of last email sync"""
        if os.path.exists(self.sync_file):
            try:
                with open(self.sync_file, 'r') as f:
                    data = json.load(f)
                    return data.get('last_sync_date')
            except:
                return None
        return None

    def _save_sync_date(self, date_str: str):
        """Save the last sync date"""
        with open(self.sync_file, 'w') as f:
            json.dump({'last_sync_date': date_str}, f)

    def _get_email_body(self, payload: Dict) -> str:
        if payload.get('mimeType') == 'text/plain':
            data = payload.get('body', {}).get('data', '')
            if data:
                return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
        
        if 'parts' in payload:
            for part in payload['parts']:
                if part.get('mimeType') == 'text/plain':
                    data = part.get('body', {}).get('data', '')
                    if data:
                        return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
        return ""

    def _get_headers(self, payload: Dict) -> Dict:
        headers = {}
        for header in payload.get('headers', []):
            headers[header['name'].lower()] = header['value']
        return headers

    def _categorize_email(self, labels: List[str], sender: str) -> str:
        label_map = {
            'CATEGORY_PROMOTIONS': 'Promotions',
            'CATEGORY_SOCIAL': 'Social',
            'CATEGORY_UPDATES': 'Updates',
            'CATEGORY_FORUMS': 'Forums',
            'IMPORTANT': 'Important',
            'STARRED': 'Important',
        }
        
        for label in labels:
            if label in label_map:
                return label_map[label]
        
        sender_lower = sender.lower()
        if any(x in sender_lower for x in ['linkedin', 'twitter', 'instagram', 'facebook', 'tinder']):
            return 'Social'
        elif any(x in sender_lower for x in ['newsletter', 'digest', 'update', 'noreply', 'no-reply']):
            return 'Updates'
        elif any(x in sender_lower for x in ['promo', 'offer', 'deal', 'discount', 'marketing']):
            return 'Promotions'
        
        return 'Personal'

    def load_emails(self, days_back: int = 7, force_full_sync: bool = False) -> List[Document]:
        """
        Load emails incrementally.
        
        Args:
            days_back: For first sync, load last X days (default: 7)
            force_full_sync: If True, ignore last sync and do full sync
        
        Returns:
            List of Document objects
        """
        # Determine date range
        if force_full_sync:
            after_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y/%m/%d')
            print(f" Force full sync: Loading emails after {after_date}")
        else:
            last_sync = self._get_last_sync_date()
            if last_sync:
                after_date = last_sync
                print(f"📥 Incremental sync: Loading emails after {after_date}")
            else:
                # First time sync
                after_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y/%m/%d')
                print(f"📥 First sync: Loading emails after {after_date}")
        
        # Build Gmail query
        query = f'after:{after_date}'
        
        try:
            results = self.service.users().messages().list(
                userId='me', 
                maxResults=self.max_results,
                q=query
            ).execute()
            
            messages = results.get('messages', [])
            
            if not messages:
                print(f"✅ No new emails found since {after_date}")
                # Still save current date as sync point
                self._save_sync_date(datetime.now().strftime('%Y/%m/%d'))
                return []
            
            documents = []
            newest_date = after_date

            for msg_info in messages:
                msg = self.service.users().messages().get(
                    userId='me', id=msg_info['id'], format='full'
                ).execute()

                payload = msg.get('payload', {})
                headers = self._get_headers(payload)
                body = self._get_email_body(payload)
                labels = msg.get('labelIds', [])
                
                sender = headers.get('from', 'Unknown')
                category = self._categorize_email(labels, sender)
                
                # Track newest email date
                email_date = headers.get('date', '')
                if email_date:
                    try:
                        parsed_date = datetime.strptime(email_date.split('+')[0].strip(), '%a, %d %b %Y %H:%M:%S')
                        newest_date = max(newest_date, parsed_date.strftime('%Y/%m/%d'))
                    except:
                        pass
                
                metadata = {
                    'subject': headers.get('subject', 'No Subject'),
                    'from': sender,
                    'date': email_date,
                    'message_id': msg_info['id'],
                    'category': category,
                    'labels': labels
                }

                doc_content = f"""
[Category: {category}]
Subject: {headers.get('subject', 'No Subject')}
From: {sender}
Date: {email_date}

Body:
{body[:2000]}
"""
                documents.append(Document(page_content=doc_content, metadata=metadata))

            # Save sync state with newest email date
            self._save_sync_date(newest_date)
            
            print(f"✅ Loaded {len(documents)} new emails (synced until {newest_date})")
            return documents

        except Exception as e:
            print(f"❌ Error loading emails: {e}")
            return []