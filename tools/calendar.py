import datetime as dt
import os

from .google_auth import _build_service


def add_calendar_event(
    summary: str,
    start_time: str,
    end_time: str,
    description: str = "",
    location: str = "",
    timezone: str = "Asia/Kolkata",
    reminders: bool = True,
) -> str:
    """
    Create a new event on the user's Google Calendar.

    Args:
        summary: Event title, e.g. "Meeting with Sarah".
        start_time: Start datetime in ISO 8601 format, e.g. "2025-07-10T09:00:00".
        end_time: End datetime in ISO 8601 format, e.g. "2025-07-10T10:00:00".
        description: Optional description / notes for the event.
        location: Optional physical or virtual location string.
        timezone: Timezone string (default: "Asia/Kolkata").
        reminders: Whether to set default 30-minute reminder (default: True).

    Returns:
        Confirmation string with event link.
    """
    calendar_id = os.getenv("GOOGLE_CALENDAR_ID", "primary")
    cal_service = _build_service("calendar", "v3")

    event_body = {
        "summary": summary,
        "location": location,
        "description": description,
        "start": {
            "dateTime": start_time,
            "timeZone": timezone,
        },
        "end": {
            "dateTime": end_time,
            "timeZone": timezone,
        },
    }

    if reminders:
        event_body["reminders"] = {
            "useDefault": False,
            "overrides": [{"method": "popup", "minutes": 30}],
        }

    event = (
        cal_service.events().insert(calendarId=calendar_id, body=event_body).execute()
    )
    return (
        f"Event created!\n"
        f"  Title    : {summary}\n"
        f"  Start    : {start_time}\n"
        f"  End      : {end_time}\n"
        f"  Location : {location or 'N/A'}\n"
        f"  Link     : {event.get('htmlLink', 'N/A')}"
    )


def get_todays_events() -> str:
    """
    Fetch all Google Calendar events scheduled for today.

    Returns:
        A formatted string with each event's title, time range, location,
        and description. Returns 'No events today' if the calendar is clear.
    """
    today = dt.date.today()
    time_min = dt.datetime(today.year, today.month, today.day).isoformat() + "Z"
    time_max = (
        dt.datetime(today.year, today.month, today.day, 23, 59, 59).isoformat() + "Z"
    )
    calendar_id = os.getenv("GOOGLE_CALENDAR_ID", "primary")

    cal_service = _build_service("calendar", "v3")
    events_result = (
        cal_service.events()
        .list(
            calendarId=calendar_id,
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )
    events = events_result.get("items", [])

    if not events:
        return "No events today."

    lines = [f"Today's Events ({today.isoformat()}):"]
    for ev in events:
        start = ev["start"].get("dateTime", ev["start"].get("date"))
        end = ev["end"].get("dateTime", ev["end"].get("date"))
        summary = ev.get("summary", "(no title)")
        location = ev.get("location", "")
        description = ev.get("description", "")
        meet_link = ""
        # Detect Google Meet / Zoom links in the description
        if "meet.google.com" in (description + ev.get("hangoutLink", "")):
            meet_link = ev.get("hangoutLink", "")
            for line in description.split("\n"):
                if "meet.google.com" in line and not meet_link:
                    meet_link = line.strip()
        loc_str = f"  Location  : {location}" if location else ""
        desc_str = f"  Details   : {description}" if description else ""
        link_str = f"  Meet Link : {meet_link}" if meet_link else ""
        lines.append(f"\n  - {summary}")
        lines.append(f"    Time     : {start} -> {end}")
        if loc_str:
            lines.append(loc_str)
        if desc_str:
            lines.append(desc_str)
        if link_str:
            lines.append(link_str)

    return "\n".join(lines)


def find_event(query: str, max_results: int = 10) -> str:
    """
    Search Google Calendar events by a text query. Matches against event
    titles, descriptions, and locations. Uses the Calendar API's built-in
    `q` parameter for server-side search, then also searches the full
    event text for broader matching.

    Args:
        query: Search text to match against event title, description, or location.
               Examples: "meeting", "dentist", "call with Sarah", "zoom".
        max_results: Maximum number of matching events to return (default 10).

    Returns:
        A formatted string with matching events including title, time, location,
        description, and a direct link to the event in Google Calendar.
    """
    calendar_id = os.getenv("GOOGLE_CALENDAR_ID", "primary")
    today = dt.date.today()
    # Search from 1 year ago to 1 year from now for broad coverage
    time_min = dt.datetime(today.year - 1, today.month, today.day).isoformat() + "Z"
    time_max = dt.datetime(today.year + 1, today.month, today.day).isoformat() + "Z"

    cal_service = _build_service("calendar", "v3")

    # Use Calendar API's built-in q search (searches title + description)
    events_result = (
        cal_service.events()
        .list(
            calendarId=calendar_id,
            q=query,
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy="startTime",
            maxResults=max_results,
        )
        .execute()
    )
    events = events_result.get("items", [])

    if not events:
        return f'No events found matching "{query}".'

    lines = [f'Events matching "{query}" ({len(events)} found):']
    for ev in events:
        start = ev["start"].get("dateTime", ev["start"].get("date"))
        end = ev["end"].get("dateTime", ev["end"].get("date"))
        summary = ev.get("summary", "(no title)")
        location = ev.get("location", "")
        description = ev.get("description", "")
        event_link = ev.get("htmlLink", "")
        meet_link = ev.get("hangoutLink", "")

        lines.append(f"\n  - {summary}")
        lines.append(f"    Time     : {start} -> {end}")
        if location:
            lines.append(f"    Location : {location}")
        if description:
            lines.append(f"    Details  : {description}")
        if meet_link:
            lines.append(f"    Meet     : {meet_link}")
        if event_link:
            lines.append(f"    Link     : {event_link}")

    return "\n".join(lines)


def update_calendar_event(
    event_id: str,
    summary: str = "",
    start_time: str = "",
    end_time: str = "",
    description: str = "",
    location: str = "",
    timezone: str = "Asia/Kolkata",
) -> str:
    """
    Update an existing Google Calendar event. Only the fields that are provided
    (non-empty strings) will be updated; fields left as empty strings are
    preserved from the existing event.

    Args:
        event_id: The ID of the event to update (from find_event or the event link).
        summary: New event title. Leave empty to keep current.
        start_time: New start datetime in ISO 8601, e.g. "2025-07-10T09:00:00". Leave empty to keep current.
        end_time: New end datetime in ISO 8601, e.g. "2025-07-10T10:00:00". Leave empty to keep current.
        description: New description / notes. Leave empty to keep current.
        location: New location string. Leave empty to keep current.
        timezone: IANA timezone string (default: "Asia/Kolkata"). Only used if start_time or end_time is set.

    Returns:
        Confirmation string with updated event details and link.
    """
    calendar_id = os.getenv("GOOGLE_CALENDAR_ID", "primary")
    cal_service = _build_service("calendar", "v3")

    # 1. Fetch the existing event
    try:
        existing = (
            cal_service.events().get(calendarId=calendar_id, eventId=event_id).execute()
        )
    except Exception as e:
        return f"ERROR: Could not find event '{event_id}': {e}"

    # 2. Build update body - only override fields that are explicitly provided
    update_body = {}

    if summary:
        update_body["summary"] = summary
    if location:
        update_body["location"] = location
    if description:
        update_body["description"] = description
    if start_time:
        if "start" not in update_body:
            update_body["start"] = dict(existing.get("start", {}))
        update_body["start"]["dateTime"] = start_time
        update_body["start"]["timeZone"] = timezone
    if end_time:
        if "end" not in update_body:
            update_body["end"] = dict(existing.get("end", {}))
        update_body["end"]["dateTime"] = end_time
        update_body["end"]["timeZone"] = timezone

    if not update_body:
        return "No updates specified. Provide at least one field to change."

    # 3. Apply the update
    try:
        updated = (
            cal_service.events()
            .update(calendarId=calendar_id, eventId=event_id, body=update_body)
            .execute()
        )
    except Exception as e:
        return f"ERROR: Failed to update event '{event_id}': {e}"

    # 4. Build confirmation showing what changed
    changes = []
    if summary:
        changes.append(f"  Title    : {summary}")
    if start_time:
        changes.append(f"  Start    : {start_time}")
    if end_time:
        changes.append(f"  End      : {end_time}")
    if location:
        changes.append(f"  Location : {location}")
    if description:
        changes.append(f"  Details  : {description}")

    return (
        f"Event updated!\n"
        + "\n".join(changes)
        + f"\n  Link     : {updated.get('htmlLink', 'N/A')}"
    )
