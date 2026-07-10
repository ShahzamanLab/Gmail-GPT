import base64
from email.message import EmailMessage
from typing import Tuple
from src.Gmail_data_loader import GmailService

class EmailSender:
    """
    Handles sending emails using the authenticated GmailService.
    Uses direct API calls for maximum speed and reliability.
    """
    def __init__(self, gmail_service: GmailService):
        self.service = gmail_service.service

    def send_email(self, to: str, subject: str, body: str) -> Tuple[bool, str]:
        """
        Sends an email via Gmail API.
        Returns: (success: bool, message: str)
        """
        try:
            if not to or "@" not in to:
                return False, "Invalid recipient email address."

            # Construct the email
            message = EmailMessage()
            message.set_content(body)
            message['To'] = to
            message['From'] = 'me'  # The authenticated user
            message['Subject'] = subject

            # Encode and send via Gmail API
            raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
            
            self.service.users().messages().send(
                userId="me", 
                body={'raw': raw_message}
            ).execute()
            
            return True, f"Email successfully sent to {to}!"
            
        except Exception as e:
            return False, f"Failed to send email: {str(e)}"