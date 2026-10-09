import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)

import time
import datetime
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from tools import TOOL_MAP, TOOLS
from tools.face import _get_deepface
from tts_engine_alt_2 import StreamingTTS

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")
MODEL = os.getenv("LLM_MODEL")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ── Streaming + tool execution ─────────────────────────────────────────────────


def stream_response(messages, tts: StreamingTTS | None = None) -> tuple[str, dict]:
    """Stream a response, returning (text, tool_calls_buffer).
    If tts is provided, tokens are fed into it sentence-by-sentence as they arrive.
    """
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        stream=True,
        temperature=0.7,
        max_tokens=250,  # limit output tokens
    )

    full_response = ""
    tool_calls_buffer = {}

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        if delta.content:
            print(delta.content, end="", flush=True)
            full_response += delta.content
            if tts is not None:
                tts.feed(delta.content)

        if delta.tool_calls:
            for tool_call in delta.tool_calls:
                idx = tool_call.index if tool_call.index is not None else 0
                # Some providers (e.g. Gemini) reuse index=0 for all tool
                # calls.  Detect a *new* tool by checking for a fresh id
                # at an already-occupied index.
                if (
                    tool_call.id is not None
                    and idx in tool_calls_buffer
                    and tool_calls_buffer[idx]["id"] is not None
                    and tool_calls_buffer[idx]["id"] != tool_call.id
                ):
                    idx = (
                        max(
                            (k for k in tool_calls_buffer if isinstance(k, int)),
                            default=-1,
                        )
                        + 1
                    )
                if idx not in tool_calls_buffer:
                    tool_calls_buffer[idx] = {
                        "id": tool_call.id,
                        "name": "",
                        "arguments": "",
                    }
                if tool_call.id and not tool_calls_buffer[idx]["id"]:
                    tool_calls_buffer[idx]["id"] = tool_call.id
                if tool_call.function.name:
                    tool_calls_buffer[idx]["name"] += tool_call.function.name
                if tool_call.function.arguments:
                    tool_calls_buffer[idx]["arguments"] += tool_call.function.arguments

    # Flush any remaining partial sentence
    if tts is not None and not tool_calls_buffer:
        tts.flush()
        tts.wait()

    return full_response, tool_calls_buffer


def execute_tools(tool_calls_buffer: dict) -> list[dict]:
    """Run all tool calls and return the result messages to append."""
    new_messages = []

    # Append the assistant's tool-use turn
    new_messages.append(
        {
            "role": "assistant",
            "content": None,  # explicit null for Gemini / other providers
            "tool_calls": [
                {
                    "id": t["id"],
                    "type": "function",
                    "function": {"name": t["name"], "arguments": t["arguments"]},
                }
                for t in tool_calls_buffer.values()
            ],
        }
    )

    for tool in tool_calls_buffer.values():
        fn = TOOL_MAP.get(tool["name"])
        if fn is None:
            print(f"\n⚠️ Unknown tool: {tool['name']}")
            continue

        try:
            args = json.loads(tool["arguments"])
        except json.JSONDecodeError:
            print(f"\n⚠️ Failed to parse arguments for {tool['name']}")
            continue

        result = fn(**args)

        # ── camera frame: send image as vision + JSON context as text ─────────
        if tool["name"] == "capture_camera_frame":
            image_b64 = result.pop("image_b64")
            context_json = json.dumps(result, indent=2, default=str)
            print(
                f"\n[Tool: capture_camera_frame] → face_count={result['face_count']}, identity={result['identity']}"
            )

            # Pass the structured data back as the tool result string
            new_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool["id"],
                    "name": tool["name"],
                    "content": (
                        "Camera frame captured. DeepFace analysis and identity lookup results:\n"
                        f"{context_json}\n\n"
                        "Use the identity.face_id if you need to register a name via register_face_name."
                    ),
                }
            )

            # To allow the LLM to analyze the image (vision), we must pass the image in a user message.
            # Tool messages strictly require string content in OpenAI/Groq specs.
            new_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Here is the visual frame captured from the camera for your analysis:",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                }
            )

        # ── generic image tool ─────────────────────────────────────────────────
        elif tool["name"] == "get_image":
            print(f"\n[Tool: get_image] → image loaded ({len(result)} base64 chars)")
            new_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool["id"],
                    "name": tool["name"],
                    "content": "Image loaded successfully. See the attached user message for the image.",
                }
            )
            new_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Here is the requested image:",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{result}"},
                        },
                    ],
                }
            )

        # ── all other tools: plain text ────────────────────────────────────────
        else:
            print(f"\n[Tool: {tool['name']}({args})] → {result}")
            new_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool["id"],
                    "name": tool["name"],
                    "content": str(result),
                }
            )

    return new_messages


# ── Chatbot ────────────────────────────────────────────────────────────────────

# SYSTEM_PROMPT = """You are Nuro, an earnest, warm, and highly concise AI with a camera stream. Use minimal tokens.
# CRITICAL: DO NOT use internal reasoning, Chain of Thought, or <thought> tags. Output ONLY the final response.
# 1. When you need to call tools, ALWAYS say a brief natural acknowledgment FIRST before the tool calls or capture_camera_frame so the user hears something immediately. Examples: "Sure, let me check.", "One moment, pulling that up.", "Let me grab your schedule." Keep it under one sentence. This is especially important when calling multiple tools like for daily briefings.
# 2. Camera: ALWAYS call `capture_camera_frame` on the first turn. Greet known users warmly by name.
# 3. New/Familiar: Welcome them and politely ask/register their name.
# 4. Memory: CRITICAL - If a user says "you know me", mentions their line of work, or implies past context, ALWAYS call `find_memory` first with broad queries like "[Name]'s work" or "[Name]'s background".
# 5. Call `save_to_memory` to store new facts/preferences (include face_id/name).
# 6. Emotion: Respond naturally to their detected mood. Never expose raw JSON or face IDs.
# 7. Daily Briefing: If user asks for their agenda, day plan, or to start their day, call: get_weather, get_news, fetch_todays_emails, get_todays_events, get_todays_tasks, get_pending_tasks. Return a very concise summary.
# 8. Formatting: Your output is spoken aloud via TTS, NOT displayed visually. NEVER use asterisks, hashes, bullet markers, dashes as list markers, markdown, or any special formatting characters. Write in plain, natural spoken English only. Use commas and periods for structure. Write numbers and units as spoken words when short (e.g. "thirty five degrees Celsius", "Artemis the second").
# """

SYSTEM_PROMPT = """You are Nuro, an earnest, warm, and concise AI with a camera stream. Use minimal tokens. Do not think out loud. Do not use <think> or <thought> tags.
1. Acknowledgments: Before the FIRST tool call in a turn, say one brief phrase. Never repeat an acknowledgment after tool results come back - just respond directly.
2. Camera & Greetings: On turn 1, capture silently, then greet known users by name. For new users, welcome them and ask their name.
3. Memory: If a user implies past context or says "you know me", call `find_memory` first with broad queries like "[Name]'s background". Call `save_to_memory` to store new facts, always including name or face_id.
4. Emotion: Respond naturally to detected mood. Never expose raw JSON or face IDs.
5. Daily Briefing: If asked for agenda, day plan, or to start the day, say a brief acknowledgment then call: get_weather, get_news, fetch_todays_emails, get_todays_events, get_todays_tasks, get_pending_tasks. Return a concise spoken summary.
6. Formatting: Output is spoken via TTS. No asterisks, hashes, dashes, bullet markers, or markdown. Plain spoken English only. Spell out short numbers and units.
"""


class Chatbot:
    def __init__(self):
        print("[Chatbot] init", flush=True)
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def chat(self, user_input: str):
        current_time = datetime.datetime.now().strftime("%I:%M %p, %A, %B %d, %Y")
        structured_input = f"[Current Time: {current_time}]\n{user_input}"
        self.messages.append({"role": "user", "content": structured_input})

        # Memory Window: Keep only the system prompt + last 10 messages (5 turns)
        # That drastically saves tokens by not repeating endless history.
        if len(self.messages) > 11:
            self.messages = [self.messages[0]] + self.messages[-10:]

        # One StreamingTTS instance per chat turn - reused across the agentic loop
        # tts.feed() is only called when delta.content tokens arrive, so tool-call-only
        # turns (like capture_camera_frame) stay silent automatically.
        tts = StreamingTTS()

        print("Assistant:", end=" ", flush=True)
        text, tool_calls = stream_response(self.messages, tts=tts)

        # Agentic loop: keep executing tools until the model stops calling them
        while tool_calls:
            tool_messages = execute_tools(tool_calls)
            self.messages.extend(tool_messages)

            print("\nAssistant:", end=" ", flush=True)
            text, tool_calls = stream_response(self.messages, tts=tts)

        self.messages.append({"role": "assistant", "content": text})
        print("\n")

        tts.stop()


if __name__ == "__main__":
    print("Starting...", flush=True)

    # ── Eager initialization: load heavy models at startup ─────────────────────
    print("[Startup] Pre-loading DeepFace model...", flush=True)
    _get_deepface()
    print("[Startup] Pre-loading TTS engine (Kokoro)...", flush=True)
    from tts_engine_alt_2 import warmup as tts_warmup

    tts_warmup()
    print("[Startup] All models loaded.\n", flush=True)

    bot = Chatbot()
    print("ARIA online. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            break
        bot.chat(user_input)
