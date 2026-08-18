import os
from typing import Any

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is not set.")


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def web_search(query: str, max_results: int = 3) -> str:
    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": TAVILY_API_KEY,
            "query": query,
            "search_depth": "basic",
            "max_results": 3,
            "include_answer": True,
        },
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    results = []

    if data.get("answer"):
        results.append(
            f"Search summary:\n{data['answer'][:2000]}"
        )

    for index, result in enumerate(
        data.get("results", [])[:3],
        start=1,
    ):
        results.append(
            f"""
Source {index}
Title: {result.get("title", "")}
URL: {result.get("url", "")}
Content:
{result.get("content", "")[:2500]}
""".strip()
        )

    return "\n\n".join(results)


def fetch_page(url: str) -> str:
    response = requests.get(
        url,
        timeout=20,
        headers={
            "User-Agent": "Mozilla/5.0 ResearchAgent/1.0"
        },
    )

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for element in soup(
        ["script", "style", "nav", "footer", "header", "aside", "noscript"]
    ):
        element.decompose()

    text = soup.get_text(separator="\n")

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    text = "\n".join(lines)

    # Keep tool output small.
    return text[:6000]


# ---------------------------------------------------------------------------
# Groq tool definitions
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information. "
                "Use this when you need factual information, "
                "recent events, research, or multiple sources."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of search results.",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": (
                "Fetch and read a webpage. "
                "Use this when you need detailed information "
                "from a specific URL."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the webpage to read.",
                    }
                },
                "required": ["url"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

TOOL_MAP = {
    "web_search": web_search,
    "fetch_page": fetch_page,
}


def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """
    Execute a tool requested by the LLM.
    """

    tool = TOOL_MAP.get(name)

    if tool is None:
        return f"Error: unknown tool '{name}'."

    try:
        result = tool(**arguments)

        return str(result)

    except requests.RequestException as exc:
        return f"Tool '{name}' failed: {exc}"

    except Exception as exc:
        return f"Tool '{name}' failed: {exc}"

