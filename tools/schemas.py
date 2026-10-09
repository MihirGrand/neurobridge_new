"""
Tool schemas (TOOLS list) and TOOL_MAP for the agent.

All tool implementations live in their own modules under tools/.
This module aggregates them so main_3.py only needs:

    from tools.schemas import TOOLS, TOOL_MAP
"""

from .calendar import (
    add_calendar_event,
    find_event,
    get_todays_events,
    update_calendar_event,
)
from .face import capture_camera_frame, get_image, register_face_name
from .gdocs import create_google_doc, read_google_doc, search_google_docs, update_google_doc
from .gmail import (
    draft_email,
    draft_reply_email,
    fetch_todays_emails,
    reply_email,
    search_emails,
    send_email,
)
from .memory import find_memory, save_to_memory
from .news import get_news
from .tasks import (
    add_google_task,
    find_task,
    get_pending_tasks,
    get_todays_tasks,
    update_google_task,
)
from .weather import get_weather
from .servo import set_horizontal_angle, set_vertical_angle
from .timers import get_current_time, set_timer, get_timers

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current time and date.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_timer",
            "description": "Set a new timer specifying the duration in minutes and a label. Note: If the user provides a time without unit (e.g. 5), assume minutes. If they provide seconds, convert to minutes (e.g. 45 seconds = 0.75 minutes).",
            "parameters": {
                "type": "object",
                "properties": {
                    "minutes": {
                        "type": "number",
                        "description": "The duration of the timer in minutes. Use fractions for seconds (e.g., 0.5 for 30 seconds).",
                    },
                    "label": {
                        "type": "string",
                        "description": "A short label or description of what the timer is for (e.g., 'Laundry', 'Pasta').",
                    }
                },
                "required": ["minutes", "label"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_timers",
            "description": "Get the list of all currently active timers, including their labels and remaining duration.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_horizontal_angle",
            "description": "Rotate the camera horizontally by providing an angle (e.g. 0-180) or 'reset' to center.",
            "parameters": {
                "type": "object",
                "properties": {
                    "angle": {
                        "type": ["integer", "string"],
                        "description": "The angle to rotate horizontally, or 'reset' to look straight.",
                    }
                },
                "required": ["angle"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_vertical_angle",
            "description": "Rotate the camera vertically by providing an angle (e.g. 0-180) or 'reset' to center.",
            "parameters": {
                "type": "object",
                "properties": {
                    "angle": {
                        "type": ["integer", "string"],
                        "description": "The angle to rotate vertically, or 'reset' to look straight.",
                    }
                },
                "required": ["angle"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get the weather for.",
                    }
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": (
                "Fetch the latest top headlines from around the world and from India using the "
                "Currents API. Supports categories like general, business, technology, sports, "
                "entertainment, health, science, politics, environment, education, and more. "
                "Use this whenever the user asks 'what's in the news', 'latest headlines', 'India news', "
                "'world news', or any current-events question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": [
                            "general",
                            "business",
                            "technology",
                            "sports",
                            "entertainment",
                            "health",
                            "science",
                            "politics_government",
                            "economy_business_finance",
                            "science_technology",
                            "arts_culture_entertainment",
                            "lifestyle_leisure",
                            "sport",
                            "crime_law_justice",
                            "education",
                            "environment",
                            "society",
                            "human_interest",
                            "labour",
                            "automotive",
                            "real_estate",
                        ],
                        "description": (
                            "The news category to fetch. Defaults to 'general' if not specified. "
                            "Common aliases also work: 'business', 'technology', 'sports', "
                            "'entertainment', 'health', 'science'."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    # -- GMAIL --
    {
        "type": "function",
        "function": {
            "name": "fetch_todays_emails",
            "description": (
                "Fetch all Gmail messages received today, returning sender, subject, date, message_id, and a "
                "short preview snippet for each email. message_id is needed if the user wants to reply. "
                "Use this when the user asks 'any new emails', 'check my inbox', 'what did I get today', or 'show me today\'s emails'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of emails to return (default 10).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_emails",
            "description": (
                "Search the user's Gmail using a free-text query. Supports the same syntax as the "
                "Gmail search box - examples: 'meeting tomorrow', 'from:john@example.com', "
                "'subject:invoice', 'has:attachment', 'label:work'. Returns sender, subject, date, "
                "message_id, and a preview snippet for each match. The message_id is needed to reply. "
                "Use this whenever the user wants to find a specific email or search their inbox."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Gmail search query - supports operators like from:, to:, subject:, "
                            "has:, label:, after:, before:, etc."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of emails to return (default 10).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": (
                "Compose and send a new email from the user's Gmail account. "
                "Use this when the user wants to send, write, or compose an email to someone. "
                "Always confirm recipient, subject, and body with the user before sending."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email address. Comma-separate for multiple.",
                    },
                    "subject": {
                        "type": "string",
                        "description": "The email subject line.",
                    },
                    "body": {
                        "type": "string",
                        "description": "The plain-text email body.",
                    },
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reply_email",
            "description": (
                "Reply to an existing Gmail email thread. The reply is automatically placed in the "
                "same thread. Requires the message_id of the email to reply to - get this from "
                "fetch_todays_emails or search_emails. Use this when the user says 'reply to that', "
                "'respond to the email from X', or 'reply to the [subject] email'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message_id": {
                        "type": "string",
                        "description": (
                            "The Gmail message ID to reply to. Get this from fetch_todays_emails "
                            "or search_emails output (the 'message_id' field)."
                        ),
                    },
                    "body": {
                        "type": "string",
                        "description": "The plain-text reply body.",
                    },
                },
                "required": ["message_id", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draft_email",
            "description": (
                "Save a new email as a Gmail Draft WITHOUT sending it. "
                "Use this when the user wants to compose an email but review it first, "
                "or says 'draft an email', 'prepare an email draft', or 'save as draft'. "
                "The draft appears in Gmail Drafts and can be reviewed or sent later."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email address. Comma-separate for multiple.",
                    },
                    "subject": {
                        "type": "string",
                        "description": "The email subject line.",
                    },
                    "body": {
                        "type": "string",
                        "description": "The plain-text email body.",
                    },
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draft_reply_email",
            "description": (
                "Save a reply to an existing Gmail thread as a Draft WITHOUT sending it. "
                "The draft is threaded under the original conversation. "
                "Use when the user wants to prepare a reply to review before sending, "
                "or says 'draft a reply', 'prepare a reply draft'. "
                "Requires the message_id from fetch_todays_emails or search_emails."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message_id": {
                        "type": "string",
                        "description": (
                            "The Gmail message ID to reply to. Get from fetch_todays_emails "
                            "or search_emails output (the 'message_id' field)."
                        ),
                    },
                    "body": {
                        "type": "string",
                        "description": "The plain-text reply body.",
                    },
                },
                "required": ["message_id", "body"],
            },
        },
    },
    # --------------
    # -- CALENDAR --

    {
        "type": "function",
        "function": {
            "name": "add_calendar_event",
            "description": (
                "Create a new event on the user's Google Calendar with a title, start/end time, "
                "optional description, location, and timezone. A 30-minute popup reminder is set by "
                "default. Use this whenever the user wants to schedule something, set up a meeting, "
                "add an event, or block time on their calendar."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "The event title, e.g. 'Meeting with Sarah'.",
                    },
                    "start_time": {
                        "type": "string",
                        "description": (
                            "Start datetime in ISO 8601 format, e.g. '2025-07-10T09:00:00'."
                        ),
                    },
                    "end_time": {
                        "type": "string",
                        "description": (
                            "End datetime in ISO 8601 format, e.g. '2025-07-10T10:00:00'."
                        ),
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional notes or agenda for the event.",
                    },
                    "location": {
                        "type": "string",
                        "description": "Optional physical address or virtual meeting link.",
                    },
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone string. Defaults to 'Asia/Kolkata'.",
                    },
                },
                "required": ["summary", "start_time", "end_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_todays_events",
            "description": (
                "Fetch all Google Calendar events scheduled for today. Returns each event's title, "
                "time range, location, description, and any video call links (Google Meet, Zoom). "
                "Use this when the user asks 'what do I have today', 'any meetings', 'my schedule', "
                "'what's on my calendar', or specifically about today's events."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_event",
            "description": (
                "Search Google Calendar events by a text query. Matches against event titles, "
                "descriptions, and locations. Searches events from the past year to one year ahead. "
                "Returns matching events with title, time, location, description, and Google Calendar "
                "links. Use this when the user wants to find a specific event, e.g. 'find my dentist "
                "appointment', 'when is the team meeting', 'search for zoom call', or 'do I have "
                "anything about taxes'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search text to match against event title, description, or location. "
                            "Examples: 'meeting', 'dentist', 'call with Sarah', 'zoom'."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of matching events to return (default 10).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_calendar_event",
            "description": (
                "Update an existing Google Calendar event. Only the fields that are provided will "
                "be changed; fields left empty are preserved from the existing event. Supports updating "
                "title, start/end time, description, and location. Requires the event ID (obtained "
                "from find_event). Use this when the user wants to reschedule, rename, move, or edit "
                "an existing event, e.g. 'move my meeting to 3pm', 'rename the dentist appointment', "
                "'change the location of the standup', or 'update the meeting agenda'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {
                        "type": "string",
                        "description": (
                            "The ID of the event to update. Get this from find_event() output."
                        ),
                    },
                    "summary": {
                        "type": "string",
                        "description": "New event title. Leave empty to keep current title.",
                    },
                    "start_time": {
                        "type": "string",
                        "description": (
                            "New start datetime in ISO 8601 format, e.g. '2025-07-10T09:00:00'. "
                            "Leave empty to keep current time."
                        ),
                    },
                    "end_time": {
                        "type": "string",
                        "description": (
                            "New end datetime in ISO 8601 format, e.g. '2025-07-10T10:00:00'. "
                            "Leave empty to keep current time."
                        ),
                    },
                    "description": {
                        "type": "string",
                        "description": "New description or agenda. Leave empty to keep current.",
                    },
                    "location": {
                        "type": "string",
                        "description": "New location. Leave empty to keep current location.",
                    },
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone string. Defaults to 'Asia/Kolkata'.",
                    },
                },
                "required": ["event_id"],
            },
        },
    },
    # --------------
    # -- TASKS --
    {
        "type": "function",
        "function": {
            "name": "add_google_task",
            "description": (
                "Create a new task in the user's Google Tasks list with a title, optional notes, "
                "and optional due date. Use this whenever the user wants to add a to-do item, "
                "remember something, create a task, or track an action item."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The task title, e.g. 'Buy groceries'.",
                    },
                    "notes": {
                        "type": "string",
                        "description": "Optional extra details or notes for the task.",
                    },
                    "due_date": {
                        "type": "string",
                        "description": "Optional due date in YYYY-MM-DD format, e.g. '2025-07-15'.",
                    },
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_google_task",
            "description": (
                "Update an existing Google Task. Only the fields that are provided will be changed; "
                "fields left empty or None are preserved from the existing task. Supports updating "
                "title, notes, due date, and completion status. Requires the task ID (obtained from "
                "find_task or add_google_task). Use this when the user wants to edit a task, mark it "
                "done, change its due date, update notes, or uncomplete a task, e.g. 'mark groceries as done', "
                "'change the report due date to Friday', 'add notes to the call mom task', or 'mark task complete'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": (
                            "The ID of the task to update. Get this from find_task() or add_google_task() output."
                        ),
                    },
                    "title": {
                        "type": "string",
                        "description": "New task title. Leave empty to keep current title.",
                    },
                    "notes": {
                        "type": "string",
                        "description": "New notes for the task. Leave empty to keep current notes.",
                    },
                    "due_date": {
                        "type": "string",
                        "description": (
                            "New due date in YYYY-MM-DD format, e.g. '2025-07-15'. "
                            "Leave empty to keep current due date."
                        ),
                    },
                    "completed": {
                        "type": "boolean",
                        "description": (
                            "Set to true to mark the task as completed, false to mark as incomplete. "
                            "Omit or leave null to keep current status unchanged."
                        ),
                    },
                },
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_todays_tasks",
            "description": (
                "Fetch all Google Tasks that are due today. Returns each task's title, notes, and "
                "due time. Only shows incomplete tasks due within today's date range. "
                "Use this when the user asks 'what do I need to do today', 'tasks for today', "
                "'what's due today', or 'today's to-dos'."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_pending_tasks",
            "description": (
                "Fetch ALL incomplete (pending) Google Tasks regardless of due date. Returns each "
                "task's title, notes, due date, and overdue status if applicable. Tasks are sorted "
                "by due date (overdue items first, then no-date items last). "
                "Use this when the user asks 'what do I have to do', 'show my tasks', 'pending work', "
                "'to-do list', 'what's pending', or 'am I overdue on anything'."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_task",
            "description": (
                "Search Google Tasks by a text query. Matches against task titles and notes "
                "(case-insensitive). Returns both pending and completed tasks with status, due date, "
                "overdue information, and notes. Use this when the user wants to find a specific task, "
                "e.g. 'find the groceries task', 'did I complete the report', 'search for buy', "
                "or 'is there a task about rent'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search text to match against task title or notes. "
                            "Examples: 'groceries', 'call mom', 'report', 'buy'."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of matching tasks to return (default 10).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    # -- GOOGLE DOCS --
    {
        "type": "function",
        "function": {
            "name": "search_google_docs",
            "description": (
                "Search the user's Google Drive for Docs matching a query. Searches both filenames and "
                "document content. Returns document titles and IDs. Use the doc_id with read_google_doc "
                "to fetch the content. Use this when the user asks to find a document, look for notes, "
                "or find something they wrote in a doc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search term, e.g. 'project proposal', 'meeting notes March'.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of docs to return (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_google_doc",
            "description": (
                "Read the full plain-text content of a Google Doc by its ID. Use this to summarise, "
                "quote from, or answer questions about a specific document. Get the doc_id from "
                "search_google_docs. Use this when the user asks 'what does that doc say', "
                "'summarise the document', or 'what's in my [name] doc'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": (
                            "Google Doc ID from search_google_docs, or extracted from the doc URL "
                            "(the part between '/d/' and '/edit')."
                        ),
                    },
                },
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_google_doc",
            "description": (
                "Create a new Google Doc with a given title and optional plain-text content. "
                "Returns the new document's ID and a direct link. Use this when the user wants to "
                "create a document, write a draft, start a new note, or make a new file in Google Docs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The title of the new document.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Optional plain-text content to write into the document body. Use \\n for paragraph breaks.",
                    },
                },
                "required": ["title", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_google_doc",
            "description": (
                "Update an existing Google Doc's content and/or sharing permissions in one call. "
                "Content: append new text to the end, or fully replace the body (append=false). "
                "Sharing: share with a specific email at reader/writer/commenter level, "
                "share publicly via link, or make the document private (remove all sharing). "
                "Use this when the user says 'add to the doc', 'update the document', "
                "'share the doc with X', 'make it public', or 'make it private'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Google Doc ID from search_google_docs or the doc URL.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Text to append or replace. Leave empty to skip content update.",
                    },
                    "append": {
                        "type": "boolean",
                        "description": "If true (default), append content at the end. If false, replace entire body.",
                    },
                    "share_with": {
                        "type": "string",
                        "description": "Email address to grant access to, e.g. 'colleague@example.com'.",
                    },
                    "share_role": {
                        "type": "string",
                        "enum": ["reader", "writer", "commenter"],
                        "description": "Permission level when sharing: 'reader' (default), 'writer', or 'commenter'.",
                    },
                    "make_public": {
                        "type": "boolean",
                        "description": "If true, anyone with the link can access at share_role level.",
                    },
                    "make_private": {
                        "type": "boolean",
                        "description": "If true, removes ALL sharing - only the owner retains access.",
                    },
                    "revoke_access_from": {
                        "type": "string",
                        "description": (
                            "Email address to remove access from specifically, e.g. 'colleague@example.com'. "
                            "Only removes that one person's access. Use make_private to remove everyone."
                        ),
                    },
                },
                "required": ["doc_id"],
            },
        },
    },

    # --------------

    {
        "type": "function",
        "function": {
            "name": "get_image",
            "description": "Retrieve and display a generic image.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "capture_camera_frame",
            "description": (
                "Capture a live frame from the camera stream. "
                "Automatically analyzes the frame for facial attributes (age, gender, emotion, race) "
                "using DeepFace, and checks the face against a local identity database to determine "
                "if the person is known, familiar (seen before but unnamed), or completely new. "
                "Use this when the user wants you to look at them, see who is there, "
                "analyze their expression or mood, or when visual context about the person is needed."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "register_face_name",
            "description": (
                "Store a person's name associated with their face in the local identity database. "
                "Call this after capturing a camera frame when the user tells you their name, "
                "or when you want to save a name for a recognized face_id. "
                "Requires the face_id returned from a prior capture_camera_frame call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "face_id": {
                        "type": "string",
                        "description": "The face_id returned from capture_camera_frame's identity block.",
                    },
                    "name": {
                        "type": "string",
                        "description": "The name to associate with this face.",
                    },
                },
                "required": ["face_id", "name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_to_memory",
            "description": (
                "Save any important information, user preferences, or facts to long-term memory "
                "in the vector database. Optionally link it to a specific face_id "
                "from the camera frame tool and their name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "info": {
                        "type": "string",
                        "description": "The piece of information to save (e.g., 'User likes coffee', 'Has a pet dog').",
                    },
                    "face_id": {
                        "type": "string",
                        "description": "The face_id of the person this info relates to, if any.",
                    },
                    "name": {
                        "type": "string",
                        "description": "The name of the person this info relates to, if any.",
                    },
                },
                "required": ["info"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_memory",
            "description": (
                "Search the long-term memory vector database for relevant information, user preferences, "
                "or facts. CRITICAL: Use this proactively whenever the user mentions themselves, "
                "their past, their preferences, or implies any personal context that you might reasonably have stored."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query (e.g., 'What did Alex ask me to remind him to buy?', 'What does Alex do?').",
                    },
                    "face_id": {
                        "type": "string",
                        "description": "Optional: Filter memories by the face_id of the person.",
                    },
                    "name": {
                        "type": "string",
                        "description": "Optional: Filter memories by the specific name of the person.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

TOOL_MAP = {
    "get_weather": get_weather,
    "get_image": get_image,
    "get_news": get_news,
    "fetch_todays_emails": fetch_todays_emails,
    "search_emails": search_emails,
    "send_email": send_email,
    "reply_email": reply_email,
    "draft_email": draft_email,
    "draft_reply_email": draft_reply_email,
    "search_google_docs": search_google_docs,
    "read_google_doc": read_google_doc,
    "create_google_doc": create_google_doc,
    "update_google_doc": update_google_doc,
    "capture_camera_frame": capture_camera_frame,
    "register_face_name": register_face_name,
    "save_to_memory": save_to_memory,
    "find_memory": find_memory,
    "add_calendar_event": add_calendar_event,
    "get_todays_events": get_todays_events,
    "find_event": find_event,
    "update_calendar_event": update_calendar_event,
    "add_google_task": add_google_task,
    "update_google_task": update_google_task,
    "get_todays_tasks": get_todays_tasks,
    "get_pending_tasks": get_pending_tasks,
    "find_task": find_task,
    "set_horizontal_angle": set_horizontal_angle,
    "set_vertical_angle": set_vertical_angle,
    "get_current_time": get_current_time,
    "set_timer": set_timer,
    "get_timers": get_timers,
}
