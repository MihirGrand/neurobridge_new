import base64
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")
MODEL = os.getenv("LLM_MODEL")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def get_weather(location: str) -> str:
    return f"Weather in {location} is sunny with a high of 25°C."


def get_image() -> str:
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image.jpg")
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


TOOLS = [
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
            "name": "get_image",
            "description": "Retrieve and display the image.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

TOOL_MAP = {
    "get_weather": get_weather,
    "get_image": get_image,
}


def stream_response(messages) -> tuple[str, dict]:
    """Stream a response, returning (text, tool_calls_buffer)."""
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        stream=True,
        temperature=0.7,
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

        if delta.tool_calls:
            for tool_call in delta.tool_calls:
                idx = tool_call.index
                if idx not in tool_calls_buffer:
                    tool_calls_buffer[idx] = {
                        "id": tool_call.id,
                        "name": "",
                        "arguments": "",
                    }
                if tool_call.function.name:
                    tool_calls_buffer[idx]["name"] += tool_call.function.name
                if tool_call.function.arguments:
                    tool_calls_buffer[idx]["arguments"] += tool_call.function.arguments

    return full_response, tool_calls_buffer


def execute_tools(tool_calls_buffer: dict) -> list[dict]:
    """Run all tool calls and return the result messages to append."""
    new_messages = []

    # Add the assistant's tool-calling message
    new_messages.append(
        {
            "role": "assistant",
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

    # Execute each tool and add its result
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

        if tool["name"] == "get_image":
            # Send as an image content block so the LLM can actually see it
            print(f"\n[Tool: get_image] → image loaded ({len(result)} base64 chars)")
            new_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool["id"],
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{result}"},
                        }
                    ],
                }
            )
        else:
            print(f"\n[Tool: {tool['name']}({args})] → {result}")
            new_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool["id"],
                    "content": result,
                }
            )

    return new_messages


class Chatbot:
    def __init__(self):
        self.messages = [
            {
                "role": "system",
                "content": "You are a helpful, concise AI assistant. Only call tools when necessary.",
            }
        ]

    def chat(self, user_input: str):
        self.messages.append({"role": "user", "content": user_input})

        print("Assistant:", end=" ", flush=True)
        text, tool_calls = stream_response(self.messages)

        if tool_calls:
            # Tool was called - execute it and do one follow-up stream
            tool_messages = execute_tools(tool_calls)
            self.messages.extend(tool_messages)

            print("Assistant:", end=" ", flush=True)
            text, _ = stream_response(self.messages)  # tools won't chain here
            self.messages.append({"role": "assistant", "content": text})
        else:
            self.messages.append({"role": "assistant", "content": text})

        print("\n")


if __name__ == "__main__":
    bot = Chatbot()
    print("Chat started (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        bot.chat(user_input)
