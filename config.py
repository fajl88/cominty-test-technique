import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_KEYS = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "llama": os.getenv("LLAMA_API_KEY")
}

BASE_URLS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "anthropic": "https://api.anthropic.com/v1/messages",
    "llama": "http://localhost:8000/v1/chat"
}
