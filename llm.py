"""LLM factory.

`OpenAILLMService` is a generic OpenAI-compatible client, so any endpoint works
via env vars — no subclass:

  - LLM_BASE_URL set   -> that endpoint (e.g. self-hosted Gemma via SGLang),
    using LLM_MODEL + LLM_API_KEY, optional LLM_TEMPERATURE.
  - LLM_BASE_URL unset -> direct OpenAI, model from OPENAI_MODEL (+ OPENAI_API_KEY).
"""

import os

from pipecat.services.openai.llm import OpenAILLMService


def build_llm(system_instruction: str, default_openai_model: str = "gpt-4o") -> OpenAILLMService:
    """Build the conversation LLM from the environment (see module docstring)."""
    base_url = os.getenv("LLM_BASE_URL")

    if base_url:
        settings: dict = {"model": os.getenv("LLM_MODEL", "google/gemma-4-26B-A4B-it")}
        temperature = os.getenv("LLM_TEMPERATURE")
        if temperature is not None:
            settings["temperature"] = float(temperature)
        return OpenAILLMService(
            base_url=base_url,
            api_key=os.getenv("LLM_API_KEY"),
            settings=OpenAILLMService.Settings(
                system_instruction=system_instruction,
                **settings,
            ),
        )

    model = os.getenv("OPENAI_MODEL", default_openai_model)
    return OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        settings=OpenAILLMService.Settings(
            model=model,
            system_instruction=system_instruction,
        ),
    )
