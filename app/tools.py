import asyncio
import requests


REQUEST_TIMEOUT = 10


def web_search(query: str):
    """
    Synchronous web search.
    """

    try:

        response = requests.get(
            "https://www.google.com/search",
            params={
                "q": query
            },
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        return response.text

    except requests.Timeout:

        return (
            "The web search timed out. "
            "Please try again with a simpler query."
        )

    except requests.RequestException as error:

        return (
            f"Web search failed: {str(error)}"
        )


async def web_search_async(query: str):
    """
    Async wrapper around the synchronous web search.
    """

    return await asyncio.to_thread(
        web_search,
        query,
    )


def fetch_page(url: str):
    """
    Synchronous webpage fetch.
    """

    try:

        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        return response.text

    except requests.Timeout:

        return (
            "Fetching the webpage timed out."
        )

    except requests.RequestException as error:

        return (
            f"Failed to fetch webpage: {str(error)}"
        )


async def fetch_page_async(url: str):
    """
    Async wrapper around the synchronous webpage fetch.
    """

    return await asyncio.to_thread(
        fetch_page,
        url,
    )


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": "Fetch the contents of a webpage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The webpage URL.",
                    }
                },
                "required": ["url"],
            },
        },
    },
]


def execute_tool(name: str, arguments: dict):

    if name == "web_search":
        return web_search(
            arguments["query"]
        )

    if name == "fetch_page":
        return fetch_page(
            arguments["url"]
        )

    return f"Unknown tool: {name}"


async def execute_tool_async(
    name: str,
    arguments: dict,
):

    if name == "web_search":
        return await web_search_async(
            arguments["query"]
        )

    if name == "fetch_page":
        return await fetch_page_async(
            arguments["url"]
        )

    return f"Unknown tool: {name}"