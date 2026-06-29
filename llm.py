"""Conversation-LLM factory.

The single seam where we choose the LLM provider — a direct port of
`../livekit-demo/fish/src/llm.py` onto Pipecat's `OpenAILLMService`.

Pipecat's `OpenAILLMService` is a generic OpenAI-compatible `/v1/chat/completions`
client: `base_url` + `api_key` are first-class constructor args, so pointing it at
our own self-hosted model needs no subclass — just a base_url. When `LLM_BASE_URL`
is set we target that endpoint (e.g. the Gemma model served via SGLang at
`https://...api.fish.audio/v1`); otherwise we fall back to direct OpenAI.

This is the SAME env contract as the LiveKit demo, so you can sub in Gemma exactly
the same way:

    LLM_BASE_URL=https://<your-sglang-host>/v1
    LLM_MODEL=google/gemma-4-26B-A4B-it
    LLM_API_KEY=<token>
    LLM_TEMPERATURE=0.6        # optional
"""

import os

from pipecat.services.openai.llm import OpenAILLMService


def build_llm(system_instruction: str, default_openai_model: str = "gpt-4o") -> OpenAILLMService:
    """Build the conversation LLM from the environment.

    - `LLM_BASE_URL` set   -> our own OpenAI-compatible endpoint, using
      `LLM_MODEL` (default Gemma) + `LLM_API_KEY`, optional `LLM_TEMPERATURE`.
    - `LLM_BASE_URL` unset  -> direct OpenAI, model from `OPENAI_MODEL` or the
      project default passed in (with `OPENAI_API_KEY`).
    """
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
