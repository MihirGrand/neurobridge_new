"""Google Docs tool - search, read, and create Google Docs via the Drive + Docs APIs."""

from .google_auth import _build_service


def search_google_docs(query: str, max_results: int = 5) -> str:
    """
    Search Google Drive for Docs matching a query string (searches filename and
    full-text content). Returns document titles and IDs.

    Args:
        query: Free-text search query, e.g. "meeting notes", "project proposal".
        max_results: Maximum number of docs to return (default 5).

    Returns:
        Formatted list of matching documents with their document IDs.
    """
    drive_service = _build_service("drive", "v3")

    # Full-text search across all Google Docs
    results = (
        drive_service.files()
        .list(
            q=f"mimeType='application/vnd.google-apps.document' and fullText contains '{query.replace(chr(39), '')}' and trashed=false",
            pageSize=max_results,
            fields="files(id, name, modifiedTime, webViewLink)",
            orderBy="modifiedTime desc",
        )
        .execute()
    )

    files = results.get("files", [])
    if not files:
        return f'No Google Docs found matching "{query}".'

    lines = [f'Found {len(files)} doc(s) matching "{query}":']
    for f in files:
        lines.append(f"\n  doc_id  : {f['id']}")
        lines.append(f"  Title   : {f['name']}")
        lines.append(f"  Modified: {f.get('modifiedTime', 'unknown')[:10]}")
        lines.append(f"  Link    : {f.get('webViewLink', 'n/a')}")

    return "\n".join(lines)


def read_google_doc(doc_id: str) -> str:
    """
    Read the full plain-text content of a Google Doc by its document ID.
    Use this to summarise, quote from, or answer questions about a specific doc.

    Args:
        doc_id: The Google Doc ID (from search_google_docs output or the URL
                between '/d/' and '/edit', e.g. '1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms').

    Returns:
        The full plain-text content of the document (may be long).
    """
    docs_service = _build_service("docs", "v1")

    doc = docs_service.documents().get(documentId=doc_id).execute()
    title = doc.get("title", "Untitled")

    # Extract text from the document body
    content = doc.get("body", {}).get("content", [])
    lines = [f"Document: {title}\n---"]

    for element in content:
        paragraph = element.get("paragraph")
        if not paragraph:
            continue
        for part in paragraph.get("elements", []):
            text_run = part.get("textRun")
            if text_run:
                text = text_run.get("content", "")
                if text.strip():
                    lines.append(text.rstrip("\n"))

    return "\n".join(lines)


def create_google_doc(title: str, content: str) -> str:
    """
    Create a new Google Doc with the given title and plain-text content.

    Args:
        title: The title for the new document.
        content: The plain-text body to write into the document. Use \\n for
                 line breaks between paragraphs.

    Returns:
        Confirmation with the new document's ID and link.
    """
    docs_service = _build_service("docs", "v1")

    # Create a blank doc
    doc = (
        docs_service.documents()
        .create(body={"title": title})
        .execute()
    )
    doc_id = doc["documentId"]

    # Insert the content if provided
    if content.strip():
        requests = [
            {
                "insertText": {
                    "location": {"index": 1},
                    "text": content,
                }
            }
        ]
        docs_service.documents().batchUpdate(
            documentId=doc_id,
            body={"requests": requests},
        ).execute()

    link = f"https://docs.google.com/document/d/{doc_id}/edit"
    return f"✅ Google Doc created: \"{title}\"\n  doc_id : {doc_id}\n  Link   : {link}"


def update_google_doc(
    doc_id: str,
    content: str = "",
    append: bool = True,
    share_with: str = "",
    share_role: str = "reader",
    make_public: bool = False,
    make_private: bool = False,
    revoke_access_from: str = "",
) -> str:
    """
    Update an existing Google Doc's content and/or sharing permissions.
    Either or both can be done in a single call.

    Content update:
        - append=True  (default): adds content at the END of the document.
        - append=False: REPLACES the entire document body with the new content.

    Sharing update (mutually exclusive - pick one):
        - share_with: grant access to a specific email at share_role level.
        - revoke_access_from: remove access from a specific email address.
        - make_public: anyone with the link can access at share_role level.
        - make_private: removes ALL non-owner permissions.

    Args:
        doc_id: The Google Doc ID (from search_google_docs or the doc URL).
        content: Text to append or use as replacement body. Leave empty to skip.
        append: If True (default), appends content. If False, replaces entire body.
        share_with: Email to grant access to, e.g. 'colleague@example.com'.
        share_role: Permission level - 'reader' (default), 'writer', or 'commenter'.
        make_public: If True, share with anyone via link at share_role level.
        make_private: If True, remove ALL sharing - only the owner retains access.
        revoke_access_from: Email address to remove access from specifically.

    Returns:
        Summary of changes made.
    """
    results = []

    # ── Content update ─────────────────────────────────────────────────────────
    if content.strip():
        docs_service = _build_service("docs", "v1")

        if append:
            # Find end index of the document to append after last character
            doc = docs_service.documents().get(documentId=doc_id).execute()
            end_index = doc["body"]["content"][-1]["endIndex"] - 1
            requests = [
                {
                    "insertText": {
                        "location": {"index": end_index},
                        "text": "\n" + content,
                    }
                }
            ]
            docs_service.documents().batchUpdate(
                documentId=doc_id, body={"requests": requests}
            ).execute()
            results.append("✅ Content appended to document.")
        else:
            # Replace: delete all body content, then insert fresh
            doc = docs_service.documents().get(documentId=doc_id).execute()
            body_content = doc["body"]["content"]
            # Find total content range (skip first structural element at index 0)
            start = body_content[0].get("endIndex", 1)
            end = body_content[-1]["endIndex"] - 1
            requests = []
            if end > start:
                requests.append(
                    {
                        "deleteContentRange": {
                            "range": {"startIndex": start, "endIndex": end}
                        }
                    }
                )
            requests.append(
                {"insertText": {"location": {"index": start}, "text": content}}
            )
            docs_service.documents().batchUpdate(
                documentId=doc_id, body={"requests": requests}
            ).execute()
            results.append("✅ Document content replaced.")

    # ── Sharing update ─────────────────────────────────────────────────────────
    if share_with or make_public or make_private or revoke_access_from:
        drive_service = _build_service("drive", "v3")

        if make_private:
            # List all non-owner permissions and delete them
            perms = (
                drive_service.permissions()
                .list(fileId=doc_id, fields="permissions(id,role)")
                .execute()
                .get("permissions", [])
            )
            removed = 0
            for perm in perms:
                if perm["role"] != "owner":
                    drive_service.permissions().delete(
                        fileId=doc_id, permissionId=perm["id"]
                    ).execute()
                    removed += 1
            results.append(f"✅ Document made private ({removed} permission(s) removed).")

        elif revoke_access_from:
            # Find the permission matching this specific email address
            perms = (
                drive_service.permissions()
                .list(
                    fileId=doc_id,
                    fields="permissions(id, role, emailAddress)",
                )
                .execute()
                .get("permissions", [])
            )
            target = next(
                (p for p in perms if p.get("emailAddress", "").lower() == revoke_access_from.lower()),
                None,
            )
            if target:
                if target["role"] == "owner":
                    results.append(f"⚠️ Cannot revoke access from {revoke_access_from} - they are the document owner.")
                else:
                    drive_service.permissions().delete(
                        fileId=doc_id, permissionId=target["id"]
                    ).execute()
                    results.append(f"✅ Access revoked from {revoke_access_from}.")
            else:
                results.append(f"⚠️ {revoke_access_from} does not have explicit access to this document.")

        elif make_public:
            drive_service.permissions().create(
                fileId=doc_id,
                body={"type": "anyone", "role": share_role},
            ).execute()
            results.append(f"✅ Document shared publicly (anyone with link: {share_role}).")

        elif share_with:
            drive_service.permissions().create(
                fileId=doc_id,
                body={"type": "user", "role": share_role, "emailAddress": share_with},
                sendNotificationEmail=True,
            ).execute()
            results.append(f"✅ Document shared with {share_with} as {share_role}.")

    if not results:
        return "⚠️ No updates made - provide content and/or a sharing change."

    link = f"https://docs.google.com/document/d/{doc_id}/edit"
    return "\n".join(results) + f"\n  Link: {link}"
