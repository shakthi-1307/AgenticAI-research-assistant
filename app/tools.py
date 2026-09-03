import asyncio
import json
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient

from .config import TAVILY_API_KEY
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
REQUEST_TIMEOUT = 10
MAX_SEARCH_RESULTS = 5


def web_search(query: str):

    try:

        print(
            f"Web search: {query}"
        )

        response = tavily_client.search(
            query=query,
            search_depth="basic",
            max_results=5,
        )

        results = response.get(
            "results",
            []
        )

        return [
            {
                "title": item.get(
                    "title",
                    ""
                ),
                "url": item.get(
                    "url",
                    ""
                ),
                "snippet": item.get(
                    "content",
                    ""
                ),
            }
            for item in results
        ]

    except Exception as error:

        return {
            "error": (
                f"Web search failed: "
                f"{error}"
            )
        }
    
def fetch_page(url: str):
    """
    Fetch a webpage and extract clean,
    readable text from the HTML.

    Raw HTML is never returned to the LLM.
    """

    try:

        response = requests.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/151.0 Safari/537.36"
                )
            },
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        # -----------------------------------------
        # Parse HTML
        # -----------------------------------------

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # -----------------------------------------
        # Remove irrelevant HTML elements
        # -----------------------------------------

        for element in soup(
            [
                "script",
                "style",
                "noscript",
                "svg",
                "nav",
                "footer",
                "header",
                "form",
                "aside",
            ]
        ):

            element.decompose()

        # -----------------------------------------
        # Extract visible text
        # -----------------------------------------

        text = soup.get_text(
            separator="\n"
        )

        # -----------------------------------------
        # Clean whitespace
        # -----------------------------------------

        lines = []

        for line in text.splitlines():

            line = line.strip()

            if line:
                lines.append(line)

        clean_text = "\n".join(lines)

        # -----------------------------------------
        # Hard safety limit
        # -----------------------------------------

        MAX_PAGE_CHARS = 12000

        clean_text = clean_text[
            :MAX_PAGE_CHARS
        ]

        return {
            "url": url,
            "content": clean_text,
        }

    except requests.Timeout:

        return {
            "error": "Fetching the webpage timed out."
        }

    except requests.RequestException as error:

        return {
            "error": (
                f"Failed to fetch webpage: "
                f"{str(error)}"
            )
        }

    except Exception as error:

        return {
            "error": (
                f"Failed to process webpage: "
                f"{str(error)}"
            )
        }
        
def calculator(expression: str):
    """
    Calculate a basic mathematical expression.
    """

    allowed_characters = (
        "0123456789+-*/(). "
    )

    if any(
        character not in allowed_characters
        for character in expression
    ):
        return {
            "error": "Invalid mathematical expression."
        }

    try:

        result = eval(
            expression,
            {
                "__builtins__": {}
            },
            {}
        )

        return {
            "expression": expression,
            "result": result,
        }

    except Exception as error:

        return {
            "error": (
                f"Could not calculate expression: "
                f"{str(error)}"
            )
        }


def get_weather(city: str):
    """
    Get current weather information for a city.
    """

    try:

        response = requests.get(
            f"https://wttr.in/{city}",
            params={
                "format": "j1"
            },
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        data = response.json()

        current = data["current_condition"][0]

        return {
            "city": city,
            "temperature_c": current["temp_C"],
            "feels_like_c": current["FeelsLikeC"],
            "humidity": current["humidity"],
            "condition": current[
                "weatherDesc"
            ][0]["value"],
            "wind_speed_kmh": current[
                "windspeedKmph"
            ],
        }

    except requests.Timeout:

        return {
            "error": "Weather request timed out."
        }

    except (
        requests.RequestException,
        KeyError,
        IndexError,
        ValueError,
    ) as error:

        return {
            "error": (
                f"Could not retrieve weather: "
                f"{str(error)}"
            )
        }


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

    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Perform basic mathematical calculations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "A mathematical expression "
                            "such as '10 * 25'."
                        ),
                    }
                },
                "required": ["expression"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get current weather information "
                "for a city."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": (
                            "The city to get weather for."
                        ),
                    }
                },
                "required": ["city"],
            },
        },
    },
]


def execute_tool(
    name: str,
    arguments: dict
):

    if name == "web_search":
        return web_search(
            arguments["query"]
        )

    if name == "fetch_page":
        return fetch_page(
            arguments["url"]
        )

    if name == "calculator":
        return calculator(
            arguments["expression"]
        )

    if name == "get_weather":
        return get_weather(
            arguments["city"]
        )

    return {
        "error": f"Unknown tool: {name}"
    }


async def execute_tool_async(
    name: str,
    arguments: dict
):
    """
    Run the blocking tool in a separate thread.
    """

    return await asyncio.to_thread(
        execute_tool,
        name,
        arguments,
    )
    
if __name__ == "__main__":

    results = web_search(
        "what is artificial intelligence"
    )

    print(
        json.dumps(
            results,
            indent=2
        )
    )