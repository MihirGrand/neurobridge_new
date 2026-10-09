"""
tools package - all agent tool functions and their schemas.

Public surface:
    from tools import TOOLS, TOOL_MAP          # for the agent loop
    from tools.face import _get_deepface       # for eager pre-loading at startup
"""

from .face import _get_deepface, capture_camera_frame, get_image, register_face_name
from .gdocs import create_google_doc, read_google_doc, search_google_docs, update_google_doc
from .gmail import (
    draft_email,
    draft_reply_email,
    fetch_todays_emails,
    reply_email,
    search_emails,
    send_email,
)
from .google_auth import setup_google_auth
from .memory import find_memory, save_to_memory
from .news import get_news
from .schemas import TOOL_MAP, TOOLS
from .weather import get_weather
from .timers import start_timer_watcher

__all__ = [
    # Agent loop
    "TOOLS",
    "TOOL_MAP",
    # Individual tools (useful for direct imports / testing)
    "get_weather",
    "get_news",
    "fetch_todays_emails",
    "search_emails",
    "send_email",
    "reply_email",
    "draft_email",
    "draft_reply_email",
    "search_google_docs",
    "read_google_doc",
    "create_google_doc",
    "update_google_doc",
    "get_image",
    "capture_camera_frame",
    "register_face_name",
    "save_to_memory",
    "find_memory",
    # Utilities
    "setup_google_auth",
    "_get_deepface",
]
