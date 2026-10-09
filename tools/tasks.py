import datetime as dt
import os

from .google_auth import _build_service


def add_google_task(
    title: str,
    notes: str = "",
    due_date: str = "",
) -> str:
    """
    Create a new task in the user's Google Tasks.

    Args:
        title: Task title, e.g. "Buy groceries".
        notes: Optional extra notes for the task.
        due_date: Optional due date in YYYY-MM-DD format, e.g. "2025-07-15".

    Returns:
        Confirmation string with task ID.
    """
    task_list_id = os.getenv("GOOGLE_TASKS_LIST_ID", "@default")
    tasks_service = _build_service("tasks", "v1")

    task_body: dict = {"title": title}
    if notes:
        task_body["notes"] = notes
    if due_date:
        try:
            parsed = dt.datetime.strptime(due_date, "%Y-%m-%d")
            task_body["due"] = parsed.strftime("%Y-%m-%dT23:59:59.000Z")
        except ValueError:
            task_body["due"] = due_date  # let the API validate

    result = (
        tasks_service.tasks().insert(tasklist=task_list_id, body=task_body).execute()
    )

    due_str = f"\n  Due  : {due_date}" if due_date else ""
    return f"Task created!\n  Title: {title}\n  ID   : {result['id']}{due_str}"


def get_todays_tasks() -> str:
    """
    Fetch all Google Tasks that are due today.

    Returns:
        A formatted string with each task's title, notes, and due time.
        Returns 'No tasks due today' if there are none.
    """
    today = dt.date.today()
    time_min = dt.datetime(today.year, today.month, today.day).isoformat() + "Z"
    time_max = (
        dt.datetime(today.year, today.month, today.day, 23, 59, 59).isoformat() + "Z"
    )
    task_list_id = os.getenv("GOOGLE_TASKS_LIST_ID", "@default")

    tasks_service = _build_service("tasks", "v1")
    tasks_result = (
        tasks_service.tasks()
        .list(
            tasklist=task_list_id,
            showCompleted=False,
            dueMin=time_min,
            dueMax=time_max,
        )
        .execute()
    )
    tasks = tasks_result.get("items", [])

    if not tasks:
        return "No tasks due today."

    lines = [f"Tasks due today ({today.isoformat()}):"]
    for t in tasks:
        title = t.get("title", "(no title)")
        due = t.get("due", "")
        notes = t.get("notes", "")
        status = t.get("status", "needsAction")
        task_id = t.get("id", "")
        note_str = f"  Notes : {notes}" if notes else ""
        due_str = f"  Due   : {due}" if due else ""
        lines.append(f"\n  [ ] {title}")
        lines.append(f"  ID    : {task_id}")
        if due_str:
            lines.append(due_str)
        if note_str:
            lines.append(note_str)

    return "\n".join(lines)


def get_pending_tasks() -> str:
    """
    Fetch all incomplete (pending) Google Tasks, regardless of due date.

    Returns:
        A formatted string with each task's title, notes, due date, and
        how overdue it is (if applicable). Returns 'No pending tasks'
        if all tasks are completed.
    """
    task_list_id = os.getenv("GOOGLE_TASKS_LIST_ID", "@default")
    today = dt.date.today()

    tasks_service = _build_service("tasks", "v1")
    tasks_result = (
        tasks_service.tasks()
        .list(
            tasklist=task_list_id,
            showCompleted=False,
            showDeleted=False,
            showHidden=False,
        )
        .execute()
    )
    tasks = tasks_result.get("items", [])

    if not tasks:
        return "No pending tasks. You're all caught up!"

    # Sort by due date (tasks without due date go to the end)
    def _sort_key(t):
        due = t.get("due", "")
        if due:
            try:
                return dt.datetime.fromisoformat(due.replace("Z", "+00:00"))
            except ValueError:
                pass
        return dt.datetime.max.replace(tzinfo=dt.timezone.utc)

    tasks.sort(key=_sort_key)

    lines = [f"Pending Tasks ({len(tasks)} total):"]
    for t in tasks:
        title = t.get("title", "(no title)")
        due = t.get("due", "")
        notes = t.get("notes", "")
        task_id = t.get("id", "")

        note_str = f"  Notes : {notes}" if notes else ""

        # Compute overdue info
        overdue_str = ""
        lines.append(f"\n  [ ] {title}")
        lines.append(f"  ID    : {task_id}")
        if due:
            try:
                due_date = dt.datetime.fromisoformat(due.replace("Z", "+00:00")).date()
                days_diff = (today - due_date).days
                if days_diff > 0:
                    overdue_str = f"  ** OVERDUE by {days_diff} day(s) **"
                lines.append(f"  Due   : {due_date.isoformat()}{overdue_str}")
            except ValueError:
                lines.append(f"  Due   : {due}")
        else:
            lines.append(f"  Due   : No due date set")

        if note_str:
            lines.append(note_str)

    return "\n".join(lines)


def find_task(query: str, max_results: int = 10) -> str:
    """
    Search Google Tasks by a text query. Matches against task titles and notes.
    Since the Google Tasks API does not support server-side text search, this
    fetches all pending tasks and filters client-side.

    Args:
        query: Search text to match against task title or notes.
               Examples: "groceries", "call mom", "report", "buy".
        max_results: Maximum number of matching tasks to return (default 10).

    Returns:
        A formatted string with matching tasks including title, notes, due date,
        status, and how overdue they are (if applicable).
    """
    task_list_id = os.getenv("GOOGLE_TASKS_LIST_ID", "@default")
    today = dt.date.today()
    query_lower = query.lower()

    tasks_service = _build_service("tasks", "v1")

    # Fetch all tasks (including completed) so we can search them
    all_tasks = []
    page_token = None
    while True:
        result = (
            tasks_service.tasks()
            .list(
                tasklist=task_list_id,
                showCompleted=True,
                showDeleted=False,
                showHidden=False,
                pageToken=page_token,
                maxResults=100,
            )
            .execute()
        )
        all_tasks.extend(result.get("items", []))
        page_token = result.get("nextPageToken")
        if not page_token:
            break

    # Filter tasks whose title or notes contain the query (case-insensitive)
    matches = []
    for t in all_tasks:
        title = t.get("title", "").lower()
        notes = t.get("notes", "").lower()
        if query_lower in title or query_lower in notes:
            matches.append(t)
        if len(matches) >= max_results:
            break

    if not matches:
        return f'No tasks found matching "{query}".'

    lines = [f'Tasks matching "{query}" ({len(matches)} found):']
    for t in matches:
        title = t.get("title", "(no title)")
        due = t.get("due", "")
        notes = t.get("notes", "")
        status = t.get("status", "needsAction")
        task_id = t.get("id", "")

        status_icon = "[x]" if status == "completed" else "[ ]"
        lines.append(f"\n  {status_icon} {title}")
        lines.append(f"    ID     : {task_id}")
        lines.append(
            f"    Status : {'Completed' if status == 'completed' else 'Pending'}"
        )

        if due:
            try:
                due_date = dt.datetime.fromisoformat(due.replace("Z", "+00:00")).date()
                days_diff = (today - due_date).days
                overdue_info = ""
                if status != "completed" and days_diff > 0:
                    overdue_info = f" (OVERDUE by {days_diff} day(s))"
                lines.append(f"    Due    : {due_date.isoformat()}{overdue_info}")
            except ValueError:
                lines.append(f"    Due    : {due}")
        else:
            lines.append("    Due    : No due date")

        if notes:
            lines.append(f"    Notes  : {notes}")

    return "\n".join(lines)


def update_google_task(
    task_id: str,
    title: str = "",
    notes: str = "",
    due_date: str = "",
    completed: bool | None = None,
) -> str:
    """
    Update an existing Google Task. Only the fields that are provided
    (non-empty strings, or non-None for completed) will be updated;
    fields left as defaults are preserved from the existing task.

    Args:
        task_id: The ID of the task to update (from find_task or add_google_task).
        title: New task title. Leave empty to keep current.
        notes: New notes. Leave empty to keep current.
        due_date: New due date in YYYY-MM-DD format, e.g. "2025-07-15". Leave empty to keep current.
        completed: Set to True to mark as completed, False to mark as incomplete.
                   Leave as None to keep current status unchanged.

    Returns:
        Confirmation string with updated task details.
    """
    task_list_id = os.getenv("GOOGLE_TASKS_LIST_ID", "@default")
    tasks_service = _build_service("tasks", "v1")

    # 1. Fetch the existing task
    try:
        existing = (
            tasks_service.tasks().get(tasklist=task_list_id, task=task_id).execute()
        )
    except Exception as e:
        return f"ERROR: Could not find task '{task_id}': {e}"

    # 2. Build update body - modify the existing task
    has_updates = False

    if title:
        existing["title"] = title
        has_updates = True
    if notes:
        existing["notes"] = notes
        has_updates = True
    if due_date:
        try:
            parsed = dt.datetime.strptime(due_date, "%Y-%m-%d")
            existing["due"] = parsed.strftime("%Y-%m-%dT23:59:59.000Z")
        except ValueError:
            existing["due"] = due_date  # let the API validate
        has_updates = True
    if completed is True:
        existing["status"] = "completed"
        has_updates = True
    elif completed is False:
        existing["status"] = "needsAction"
        has_updates = True

    if not has_updates:
        return "No updates specified. Provide at least one field to change."

    # 3. Apply the update
    try:
        updated = (
            tasks_service.tasks()
            .update(tasklist=task_list_id, task=task_id, body=existing)
            .execute()
        )
    except Exception as e:
        return f"ERROR: Failed to update task '{task_id}': {e}"

    # 4. Build confirmation showing what changed
    changes = [f"Task updated!"]
    if title:
        changes.append(f"  Title : {title}")
    if notes:
        changes.append(f"  Notes : {notes}")
    if due_date:
        changes.append(f"  Due   : {due_date}")
    if completed is True:
        changes.append(f"  Status: Marked as COMPLETED")
    elif completed is False:
        changes.append(f"  Status: Marked as PENDING")
    changes.append(f"  ID    : {task_id}")

    return "\n".join(changes)
