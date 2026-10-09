"""Memory tool - save and search long-term memories via ChromaDB."""

import uuid

import chromadb
from pathlib import Path

# ── ChromaDB setup (memory collection) ────────────────────────────────────────
CHROMA_DIR = Path("./face_db")
CHROMA_DIR.mkdir(exist_ok=True)

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
memory_collection = chroma_client.get_or_create_collection(
    name="memory",
    metadata={"hnsw:space": "cosine"},
)


def save_to_memory(info: str, face_id: str = None, name: str = None) -> str:
    """
    Save any relevant information to the LLM's long-term memory vector database.
    Optionally link it to a face_id and name if it's about a specific person.
    """
    doc_id = str(uuid.uuid4())
    metadata = {}
    if face_id:
        metadata["face_id"] = face_id
    if name:
        metadata["name"] = name

    try:
        memory_collection.add(
            documents=[info], metadatas=[metadata] if metadata else None, ids=[doc_id]
        )
        print(f"[MemoryDB] Saved info for face_id={face_id}, name={name}: {info}")
        return f"Successfully saved memory: {info}"
    except Exception as e:
        return f"Failed to save memory: {e}"


def find_memory(query: str, face_id: str = None, name: str = None) -> list[str]:
    """
    Search the LLM's long-term memory vector database for relevant information.
    Optionally filter by face_id and/or name.
    """
    where_filter = None
    if face_id and name:
        where_filter = {"$and": [{"face_id": face_id}, {"name": name}]}
    elif face_id:
        where_filter = {"face_id": face_id}
    elif name:
        where_filter = {"name": name}

    try:
        results = memory_collection.query(
            query_texts=[query],
            n_results=5,
            where=where_filter,
        )
        if results and results.get("documents") and results["documents"][0]:
            found_memories = results["documents"][0]
            print(
                f"[MemoryDB] Found {len(found_memories)} memories for query '{query}'"
            )
            return found_memories
        else:
            print(f"[MemoryDB] No memories found for query '{query}'")
            return ["No relevant memories found."]
    except Exception as e:
        return [f"Failed to search memory: {e}"]
