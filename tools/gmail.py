"""Gmail tool - fetch, search, send, and reply to emails via the Gmail API."""

import base64
import datetime as dt
import email.mime.text

from .google_auth import _build_service


def fetch_todays_emails(max_results: int = 10) -> str:
    """
    Fetch all Gmail messages received today, including sender, subject,
    snippet, and date.

    Args:
        max_results: Maximum number of emails to return (default 10).

    Returns:
        Formatted list of today's emails.
    """
    gmail_service = _build_service("gmail", "v1")
    user_id = "me"

    today = dt.date.today()
    after = f"{today.isoformat()}"

    messages = (
        gmail_service.users()
        .messages()
        .list(userId=user_id, q=f"after:{after}", maxResults=max_results)
        .execute()
        .get("messages", [])
    )

    if not messages:
        return f"📭 No emails received today ({after})."

    lines = [f"📬 Emails received today ({len(messages)} shown):"]
    for msg_meta in messages:
        msg = (
            gmail_service.users()
            .messages()
            .get(
                userId=user_id,
                id=msg_meta["id"],
                format="metadata",
                metadataHeaders=["From", "Subject", "Date"],
            )
            .execute()
        )
        headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
        frm = headers.get("From", "Unknown")
        subject = headers.get("Subject", "(no subject)")
        date = headers.get("Date", "")
        snippet = msg.get("snippet", "")

        lines.append(f"\n  message_id: {msg_meta['id']}")
        lines.append(f"  📧 From    : {frm}")
        lines.append(f"     Subject : {subject}")
        lines.append(f"     Date    : {date}")
        lines.append(f"     Preview : {snippet}")

    return "\n".join(lines)


def search_emails(query: str, max_results: int = 10) -> str:
    """
    Search the user's Gmail using a free-text query string (same syntax as the
    Gmail search box).

    Args:
        query: Gmail search query, e.g. "meeting tomorrow", "from:john@example.com",
               "subject:invoice", "has:attachment".
        max_results: Maximum number of results to return (default 10).

    Returns:
        Formatted list of matching emails with their message IDs (needed for reply_email).
    """
    gmail_service = _build_service("gmail", "v1")
    user_id = "me"

    messages = (
        gmail_service.users()
        .messages()
        .list(userId=user_id, q=query, maxResults=max_results)
        .execute()
        .get("messages", [])
    )

    if not messages:
        return f'No emails found for query: "{query}"'

    lines = [f'Search results for "{query}" ({len(messages)} shown):']
    for msg_meta in messages:
        msg = (
            gmail_service.users()
            .messages()
            .get(
                userId=user_id,
                id=msg_meta["id"],
                format="metadata",
                metadataHeaders=["From", "Subject", "Date"],
            )
            .execute()
        )
        headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
        frm = headers.get("From", "Unknown")
        subject = headers.get("Subject", "(no subject)")
        date = headers.get("Date", "")
        snippet = msg.get("snippet", "")

        lines.append(f"\n  message_id: {msg_meta['id']}")
        lines.append(f"  From    : {frm}")
        lines.append(f"  Subject : {subject}")
        lines.append(f"  Date    : {date}")
        lines.append(f"  Preview : {snippet}")

    return "\n".join(lines)


def send_email(to: str, subject: str, body: str) -> str:
    """
    Compose and send a new email from the authenticated Gmail account.

    Args:
        to: Recipient email address (or comma-separated list for multiple recipients).
        subject: Email subject line.
        body: Plain-text email body.

    Returns:
        Confirmation string with the sent message ID.
    """
    gmail_service = _build_service("gmail", "v1")

    msg = email.mime.text.MIMEText(body, "plain")
    msg["To"] = to
    msg["Subject"] = subject

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    result = (
        gmail_service.users()
        .messages()
        .send(userId="me", body={"raw": raw})
        .execute()
    )
    return f"✅ Email sent to {to}. Message ID: {result['id']}"


def reply_email(message_id: str, body: str) -> str:
    """
    Reply to an existing Gmail thread, preserving thread context via proper
    In-Reply-To and References headers so it appears in the same thread.

    Args:
        message_id: The Gmail message ID to reply to. Obtain from search_emails
                    or fetch_todays_emails output (the 'message_id' field).
        body: Plain-text reply body.

    Returns:
        Confirmation string with the sent message ID.
    """
    gmail_service = _build_service("gmail", "v1")
    user_id = "me"

    # Fetch the original message to extract threading headers
    original = (
        gmail_service.users()
        .messages()
        .get(
            userId=user_id,
            id=message_id,
            format="metadata",
            metadataHeaders=["From", "Subject", "Message-ID", "References"],
        )
        .execute()
    )
    headers = {h["name"]: h["value"] for h in original["payload"]["headers"]}
    thread_id = original["threadId"]

    reply_to = headers.get("From", "")
    subject = headers.get("Subject", "")
    if not subject.lower().startswith("re:"):
        subject = f"Re: {subject}"
    orig_message_id = headers.get("Message-ID", "")
    references = headers.get("References", "")
    if orig_message_id:
        references = (references + " " + orig_message_id).strip()

    msg = email.mime.text.MIMEText(body, "plain")
    msg["To"] = reply_to
    msg["Subject"] = subject
    if orig_message_id:
        msg["In-Reply-To"] = orig_message_id
    if references:
        msg["References"] = references

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    result = (
        gmail_service.users()
        .messages()
        .send(userId=user_id, body={"raw": raw, "threadId": thread_id})
        .execute()
    )
    return f"✅ Reply sent to {reply_to}. Message ID: {result['id']}"


def draft_email(to: str, subject: str, body: str) -> str:
    """
    Save a new email as a Gmail Draft (does NOT send it).
    The user can review and send it manually, or ask the agent to send it later.

    Args:
        to: Recipient email address (or comma-separated list).
        subject: Email subject line.
        body: Plain-text email body.

    Returns:
        Confirmation string with the draft ID.
    """
    gmail_service = _build_service("gmail", "v1")

    msg = email.mime.text.MIMEText(body, "plain")
    msg["To"] = to
    msg["Subject"] = subject

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    result = (
        gmail_service.users()
        .drafts()
        .create(userId="me", body={"message": {"raw": raw}})
        .execute()
    )
    return f"✅ Draft saved (not sent). Draft ID: {result['id']}. To: {to} | Subject: {subject}"


def draft_reply_email(message_id: str, body: str) -> str:
    """
    Save a reply as a Gmail Draft in the same thread as the original message.
    The draft is NOT sent - the user can review it and send it manually or via the agent.

    Args:
        message_id: The Gmail message ID to reply to (from fetch_todays_emails or
                    search_emails output - the 'message_id' field).
        body: Plain-text reply body.

    Returns:
        Confirmation string with the draft ID.
    """
    gmail_service = _build_service("gmail", "v1")
    user_id = "me"

    # Fetch the original message to extract threading headers
    original = (
        gmail_service.users()
        .messages()
        .get(
            userId=user_id,
            id=message_id,
            format="metadata",
            metadataHeaders=["From", "Subject", "Message-ID", "References"],
        )
        .execute()
    )
    headers = {h["name"]: h["value"] for h in original["payload"]["headers"]}
    thread_id = original["threadId"]

    reply_to = headers.get("From", "")
    subject = headers.get("Subject", "")
    if not subject.lower().startswith("re:"):
        subject = f"Re: {subject}"
    orig_message_id = headers.get("Message-ID", "")
    references = headers.get("References", "")
    if orig_message_id:
        references = (references + " " + orig_message_id).strip()

    msg = email.mime.text.MIMEText(body, "plain")
    msg["To"] = reply_to
    msg["Subject"] = subject
    if orig_message_id:
        msg["In-Reply-To"] = orig_message_id
    if references:
        msg["References"] = references

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    result = (
        gmail_service.users()
        .drafts()
        .create(
            userId=user_id,
            body={"message": {"raw": raw, "threadId": thread_id}},
        )
        .execute()
    )
    return f"✅ Reply draft saved (not sent). Draft ID: {result['id']}. To: {reply_to} | Subject: {subject}"
