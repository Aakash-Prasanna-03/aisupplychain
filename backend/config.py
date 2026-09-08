import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

# Qwen is the default real-agent provider. It uses the standard
# OpenAI-compatible /chat/completions contract. A local Qwen server can be
# used by pointing QWEN_BASE_URL at its compatible endpoint.
LLM_ENABLED = os.getenv("LLM_ENABLED", "false").lower() == "true"
LLM_API_KEY = os.getenv("QWEN_API_KEY", os.getenv("LLM_API_KEY", ""))
LLM_BASE_URL = os.getenv("QWEN_BASE_URL", os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
LLM_MODEL = os.getenv("QWEN_MODEL", os.getenv("LLM_MODEL", "qwen3.5-plus"))
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
LLM_LOCAL_ENDPOINT = LLM_BASE_URL.startswith(("http://localhost", "http://127.0.0.1"))
LLM_CONFIGURED = LLM_ENABLED and bool(LLM_API_KEY or LLM_LOCAL_ENDPOINT)
MIN_SERVICE_LEVEL = float(os.getenv("MIN_SERVICE_LEVEL", "0.70"))
FAIRNESS_THRESHOLD = float(os.getenv("FAIRNESS_THRESHOLD", "0.08"))
