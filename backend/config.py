import os

LLM_ENABLED = os.getenv("LLM_ENABLED", "false").lower() == "true"
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
MIN_SERVICE_LEVEL = float(os.getenv("MIN_SERVICE_LEVEL", "0.70"))
FAIRNESS_THRESHOLD = float(os.getenv("FAIRNESS_THRESHOLD", "0.08"))
