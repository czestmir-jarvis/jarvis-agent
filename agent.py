"""Jarvis — Claude as a voice-first AI assistant with memory and web search."""
from __future__ import annotations
import json, re
from pathlib import Path
from anthropic import Anthropic
from config import settings

MEMORY_FILE = Path(__file__).parent / "jarvis_memory.json"

SYSTEM_PROMPT = """\
You are Jarvis — an advanced AI assistant modeled after the AI from Iron Man. \
You are loyal, intelligent, witty, and resourceful. You address the user as "sir" \
and speak in a refined British manner — concise, confident, and occasionally dry-humored.

Your capabilities:
- Conversing naturally on any topic with deep knowledge
- Searching the web for current information when needed
- Remembering context from previous conversations
- Providing analysis, recommendations, and creative solutions

Behavioral rules:
- Keep responses concise — you are SPEAKING aloud, not writing essays. 2-4 sentences is ideal.
- Never use markdown formatting (no **, ##, ```, bullet points). Speak in plain natural language.
- Be proactive — anticipate needs, offer relevant follow-ups.
- If you don't know something current, use the web_search tool.
- You have personality. Be warm but professional, occasionally witty.
"""

TOOLS = [{
    "name": "web_search",
    "description": "Search the web for current information, news, or facts.",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Search query"}},
        "required": ["query"]
    }
}]

def _load_memory() -> list[dict]:
    if MEMORY_FILE.exists():
        try: return json.loads(MEMORY_FILE.read_text())[-100:]
        except: return []
    return []

def _save_memory(messages: list[dict]):
    try:
        mem = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] in ("user","assistant") and isinstance(m["content"], str)]
        MEMORY_FILE.write_text(json.dumps(mem[-100:], indent=2))
    except: pass

def _search(query: str) -> str:
    import httpx
    try:
        with httpx.Client(timeout=15) as c:
            r = c.get("https://html.duckduckgo.com/html/", params={"q": query}, headers={"User-Agent": "Mozilla/5.0"})
            results = []
            for i, chunk in enumerate(r.text.split("result__title")[1:8], 1):
                ts, te = chunk.find(">")+1, chunk.find("</a>")
                if ts > 0 and te > 0:
                    title = re.sub(r"<[^>]+>", "", chunk[ts:te]).strip()
                    ss = chunk.find("result__snippet")
                    snippet = ""
                    if ss > 0:
                        s1, s2 = chunk.find(">", ss)+1, chunk.find("</", chunk.find(">", ss)+1)
                        if s1 > 0 and s2 > 0: snippet = re.sub(r"<[^>]+>", "", chunk[s1:s2]).strip()
                    results.append(f"{i}. {title}\n   {snippet}")
            return "\n\n".join(results) if results else "No results found."
    except Exception as e: return f"Search failed: {e}"

async def chat(user_message: str) -> str:
    client = Anthropic(api_key=settings.anthropic_api_key)
    history = _load_memory()
    history.append({"role": "user", "content": user_message})
    messages = history[-20:]

    for _ in range(5):
        response = client.messages.create(model=settings.model, max_tokens=1024, system=SYSTEM_PROMPT, tools=TOOLS, messages=messages)
        tool_calls = [b for b in response.content if b.type == "tool_use"]
        text_parts = [b.text for b in response.content if b.type == "text"]

        if not tool_calls:
            result = " ".join(text_parts)
            history.append({"role": "assistant", "content": result})
            _save_memory(history)
            return result

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for tc in tool_calls:
            if tc.name == "web_search":
                tool_results.append({"type": "tool_result", "tool_use_id": tc.id, "content": _search(tc.input.get("query", ""))})
        messages.append({"role": "user", "content": tool_results})

    result = " ".join(text_parts) if text_parts else "I seem to have hit a snag, sir."
    history.append({"role": "assistant", "content": result})
    _save_memory(history)
    return result
