"""Jarvis backend — direct Claude conversation with memory and web search."""
from __future__ import annotations
import os
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime, timezone

from anthropic import Anthropic
import httpx

# ─── Config ────────────────────────────────────────────────────────────────────
API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MODEL = os.getenv("MODEL", "claude-sonnet-4-6")
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "50"))  # messages to keep in memory

SYSTEM_PROMPT = """\
You are Jarvis — a sophisticated AI assistant inspired by the AI from Iron Man.
You speak with a refined British manner. You address the user as "sir" naturally 
(not excessively). You are witty, capable, and efficient.

Key behaviours:
- Be concise in speech — your responses will be read aloud, so keep them natural 
  and conversational (2-4 sentences for simple queries, longer only when needed)
- When you use web_search, summarise findings conversationally — don't list URLs
- You have memory of previous conversations with this user
- You're proactive — suggest follow-ups when appropriate
- You have personality — dry humour, understated confidence, genuine helpfulness

Current date: {date}
"""

# ─── Conversation Memory ───────────────────────────────────────────────────────
MEMORY_DIR = Path("/tmp/jarvis_memory")
MEMORY_DIR.mkdir(exist_ok=True)

def _memory_path(user_id: str) -> Path:
    h = hashlib.sha256(user_id.encode()).hexdigest()[:12]
    return MEMORY_DIR / f"{h}.json"

def load_history(user_id: str) -> list[dict]:
    p = _memory_path(user_id)
    if p.exists():
        try:
            data = json.loads(p.read_text())
            return data.get("messages", [])[-MAX_HISTORY:]
        except Exception:
            return []
    return []

def save_history(user_id: str, messages: list[dict]):
    p = _memory_path(user_id)
    # Keep summary of older messages + recent messages
    trimmed = messages[-MAX_HISTORY:]
    p.write_text(json.dumps({
        "user_id": user_id,
        "updated": datetime.now(timezone.utc).isoformat(),
        "messages": trimmed,
    }))


# ─── Web Search Tool ──────────────────────────────────────────────────────────
TOOLS = [
    {
        "name": "web_search",
        "description": "Search the web for current information, news, facts, or anything the user asks about. Use this whenever you need up-to-date information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    }
]

def do_web_search(query: str) -> str:
    """Search DuckDuckGo and return results."""
    try:
        with httpx.Client(timeout=12) as client:
            resp = client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                follow_redirects=True,
            )
            results = []
            parts = resp.text.split("result__title")
            for chunk in parts[1:8]:
                # Extract title
                a_start = chunk.find(">") + 1
                a_end = chunk.find("</a>")
                if a_start > 0 and a_end > a_start:
                    title = chunk[a_start:a_end].replace("<b>", "").replace("</b>", "").strip()
                    # Extract snippet
                    snip_start = chunk.find("result__snippet")
                    snippet = ""
                    if snip_start > 0:
                        s_start = chunk.find(">", snip_start) + 1
                        s_end = chunk.find("</", s_start)
                        if s_start > 0 and s_end > s_start:
                            snippet = chunk[s_start:s_end].replace("<b>", "").replace("</b>", "").strip()
                    # Extract URL
                    href_start = chunk.find('href="')
                    url = ""
                    if href_start > 0:
                        href_start += 6
                        href_end = chunk.find('"', href_start)
                        url = chunk[href_start:href_end]
                    
                    results.append(f"- {title}: {snippet}" + (f" ({url})" if url and url.startswith("http") else ""))
            
            if results:
                return "Search results:\n" + "\n".join(results)
            return "No results found for this query."
    except Exception as e:
        return f"Search failed: {str(e)}"


def execute_tool(name: str, tool_input: dict) -> str:
    if name == "web_search":
        return do_web_search(tool_input.get("query", ""))
    return f"Unknown tool: {name}"


# ─── Chat Function ─────────────────────────────────────────────────────────────
def chat(user_id: str, message: str) -> tuple[str, list[dict]]:
    """Send a message and get a response. Returns (response_text, updated_history)."""
    client = Anthropic(api_key=API_KEY)
    
    history = load_history(user_id)
    history.append({"role": "user", "content": message})
    
    system = SYSTEM_PROMPT.format(date=datetime.now(timezone.utc).strftime("%A, %d %B %Y"))
    
    # Agentic loop — keep going until Claude gives a text response
    for _ in range(5):  # max 5 tool rounds
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=system,
            tools=TOOLS,
            messages=history,
        )
        
        # Collect response
        text_parts = []
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(block)
        
        if not tool_calls:
            # Done — just text response
            assistant_text = "\n".join(text_parts)
            history.append({"role": "assistant", "content": assistant_text})
            save_history(user_id, history)
            return assistant_text, history
        
        # Execute tools
        history.append({"role": "assistant", "content": response.content})
        tool_results = []
        for tc in tool_calls:
            result = execute_tool(tc.name, tc.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": result,
            })
        history.append({"role": "user", "content": tool_results})
    
    # Fallback
    fallback = "\n".join(text_parts) if text_parts else "I'm having trouble completing that request, sir."
    history.append({"role": "assistant", "content": fallback})
    save_history(user_id, history)
    return fallback, history


def clear_history(user_id: str):
    p = _memory_path(user_id)
    if p.exists():
        p.unlink()
