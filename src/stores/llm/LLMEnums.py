from enum import Enum

class LLMEnums(Enum):
    OLLAMA = "OLLAMA"
    OLLAMAE = "OLLAMAE"
    GEMINI = "GEMINI"
    GEMINIE = "GEMINIE"
    OPENAI="OPENAI"
    GROQ = "GROQ"

class OllamaEnums(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

class GeminiEnums(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    MODEL = "model"
    FUNCTION = "function"

class OpenAIEnums(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

