"""
Google OAuth helper utilities shared across all Google API tools.

Setup steps (only needed once):
  a. Go to https://console.cloud.google.com/apis/credentials
  b. Create a project (or pick existing)
  c. Create an "OAuth client ID" (Desktop app type)
  d. Download the JSON → save as credentials.json
  e. Enable APIs: Google Calendar, Tasks API, Gmail API
  f. Run setup_google_auth() → it opens a browser window for you to authorise
"""

import os

from google.auth.transport.requests import Request

# Scopes your agent needs - add more here if you expand tooling.
#   - calendar  : read & write events
#   - tasks     : read & write tasks
#   - gmail     : read & send mail, modify labels
GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/tasks",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/documents",
]

# Module-level cache so the auth flow only runs once per process.
_cached_creds = None


def _get_token_path() -> str:
    return os.getenv("GOOGLE_TOKEN_PATH", "token.json")


def _get_credentials_path() -> str:
    return os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")


def setup_google_auth() -> None:
    """
    One-time setup: authenticate with Google and save the token.
    Call this at the start of your application (or it will be called
    automatically on first use by _get_google_credentials).

    Flow:
      1. Reads OAuth client secrets from credentials.json
      2. If token.json exists → loads & refreshes it
      3. If token.json is missing/expired → opens browser for consent
      4. Saves the refreshed/new token back to token.json
    """
    global _cached_creds

    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    token_path = _get_token_path()
    creds_path = _get_credentials_path()

    creds = None

    # 1. Try loading an existing token
    if os.path.exists(token_path):
        try:
            creds = Credentials.from_authorized_user_file(token_path, GOOGLE_SCOPES)
        except Exception:
            creds = None  # corrupt or incompatible token → re-auth

    # 2. If no valid creds, run the browser-based OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            # Token expired but we have a refresh token - refresh silently
            try:
                creds.refresh(Request())
            except Exception:
                # Refresh failed (revoked, password changed, etc.) → full re-auth
                creds = None

        if not creds or not creds.valid:
            if not os.path.exists(creds_path):
                raise FileNotFoundError(
                    f"Missing '{creds_path}'.\n\n"
                    "To set up Google auth, follow these steps:\n"
                    "  1. Go to https://console.cloud.google.com/apis/credentials\n"
                    "  2. Create an OAuth 2.0 Client ID (type: Desktop app)\n"
                    "  3. Enable these APIs: Google Calendar, Tasks, Gmail\n"
                    "  4. Download the JSON and save it as 'credentials.json'\n"
                    "     (or set GOOGLE_CREDENTIALS_PATH env var)\n"
                    "  5. Run setup_google_auth() again\n"
                )

            flow = InstalledAppFlow.from_client_secrets_file(creds_path, GOOGLE_SCOPES)

            # ── IMPORTANT ──────────────────────────────────────────
            # Default run_local_server() opens a browser tab on the
            # same machine.  If your agent runs on a headless server /
            # Docker container, use run_console() instead, which prints
            # the auth URL to the terminal for you to open manually.
            # ───────────────────────────────────────────────────────
            try:
                creds = flow.run_local_server(port=0)
            except Exception:
                # Fallback: if no browser available, use copy-paste flow
                creds = flow.run_console()

    # 3. Persist token for future runs
    with open(token_path, "w") as f:
        f.write(creds.to_json())

    _cached_creds = creds
    print(f"✅ Google auth successful. Token saved to {token_path}")


def _get_google_credentials():
    """
    Return valid Google credentials, auto-running setup_google_auth()
    on the very first call if no token.json exists yet.
    """
    global _cached_creds

    if _cached_creds is not None:
        # Check if cached creds are still valid
        if _cached_creds.valid:
            return _cached_creds
        if _cached_creds.expired and _cached_creds.refresh_token:
            try:
                _cached_creds.refresh(Request())
                # Update the token file with refreshed creds
                with open(_get_token_path(), "w") as f:
                    f.write(_cached_creds.to_json())
                return _cached_creds
            except Exception:
                # Refresh failed (revoked, password changed, etc.) → re-auth
                _cached_creds = None

    # First run or token invalid → kick off the full auth flow
    setup_google_auth()
    return _cached_creds


def _build_service(api_name: str, api_version: str):
    """Return an authorised Google API service object."""
    from googleapiclient.discovery import build

    creds = _get_google_credentials()
    return build(api_name, api_version, credentials=creds)
