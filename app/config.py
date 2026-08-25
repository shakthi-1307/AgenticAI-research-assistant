import os

from dotenv import load_dotenv


load_dotenv()


GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

TAVILY_API_KEY = os.getenv(
    "TAVILY_API_KEY"
)

MODEL = os.getenv(
    "MODEL",
    "openai/gpt-oss-20b"
)