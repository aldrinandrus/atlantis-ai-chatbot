import os


def get_embeddings():
    provider = (
        os.getenv("EMBEDDING_PROVIDER") or os.getenv("LLM_PROVIDER") or "openai"
    ).lower()

    if provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency for OpenAI embeddings. Install langchain-openai."
            ) from exc

        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        return OpenAIEmbeddings(model=model, api_key=api_key)

    if provider == "gemini":
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency for Gemini embeddings. Install langchain-google-genai."
            ) from exc

        model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")
        api_key = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GEMINI_API_KEY_1")
            or os.getenv("GEMINI_API_KEY_2")
        )
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        return GoogleGenerativeAIEmbeddings(google_api_key=api_key, model=model)

    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {provider}")
