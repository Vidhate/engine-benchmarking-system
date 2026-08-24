import os

# Unit tests never touch the network: no LangSmith export, no real OpenAI key.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ.setdefault("OPENAI_API_KEY", "sk-unit-test-not-a-real-key")
