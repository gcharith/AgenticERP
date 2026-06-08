"""LLM service – unified interface to OpenAI & Anthropic with cost tracking."""

from __future__ import annotations

import logging
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class LLMService:
    def __init__(self) -> None:
        self.openai = ChatOpenAI(
            model=settings.default_llm_model,
            temperature=0,
            api_key=settings.openai_api_key,
        )
        self.anthropic = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            temperature=0,
            api_key=settings.anthropic_api_key,
        )
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )

    async def chat(self, messages: list[dict[str, str]], provider: str = "openai", **kwargs: Any) -> str:
        """Send a list of role/content messages to the chosen provider and return the reply text."""
        from langchain_core.messages import HumanMessage, SystemMessage

        # convert our simple dict messages into langchain message objects
        lc_messages = []
        for m in messages:
            if m["role"] == "system":
                lc_messages.append(SystemMessage(content=m["content"]))
            else:
                lc_messages.append(HumanMessage(content=m["content"]))

        llm = self.openai if provider == "openai" else self.anthropic
        response = llm.invoke(lc_messages)
        return response.content

    async def embed(self, text: str) -> list[float]:
        """Return the embedding vector for a single piece of text."""
        return self.embeddings.embed_query(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for a batch of texts in one call."""
        return self.embeddings.embed_documents(texts)


def get_llm_service() -> LLMService:
    return LLMService()
