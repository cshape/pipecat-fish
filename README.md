# pipecat-fish

A minimal [Pipecat](https://docs.pipecat.ai) voice agent that speaks with
[Fish Audio](https://fish.audio) TTS. You talk in the browser; it transcribes,
runs an LLM, and replies in a Fish voice.

## Stack

| Role | Service | Pipecat extra |
|------|---------|---------------|
| STT  | **Deepgram Flux** (default) or AssemblyAI | `pipecat-ai[deepgram]` / `[assemblyai]` |
| LLM  | OpenAI-compatible — `gpt-*`, or self-hosted Gemma via `LLM_BASE_URL` | `pipecat-ai[openai]` |
| TTS  | Fish Audio `s2.1-pro`, PCM @ 24 kHz, low latency | `pipecat-ai[fish]` |
| VAD  | Silero (barge-in) | `pipecat-ai[silero]` |
| Transport | Small WebRTC (local) / Daily | `pipecat-ai[webrtc,daily,runner]` |

## Quick start

```bash
cp .env.example .env.local        # fill in DEEPGRAM_API_KEY, FISH_API_KEY, and either OPENAI_API_KEY or the LLM_* (Gemma) vars
uv sync
uv run bot.py
```

Open <http://localhost:7860>, allow the mic, and talk.

## Turn-taking

End-of-turn detection depends on `STT_PROVIDER`:

- **`deepgram`** (default) — Deepgram **Flux** detects end-of-turn server-side, no
  separate turn model. Tune with `DEEPGRAM_EOT_THRESHOLD` (0.5–0.9; lower = faster,
  more false ends) and `DEEPGRAM_EOT_TIMEOUT_MS`. Silero VAD runs only for barge-in.
- **`assemblyai`** — Silero VAD gates speech; the bundled **smart-turn-v3** model
  decides end-of-turn. Tune the worst-case wait with `SMART_TURN_STOP_SECS` (default 0.6 s).

Both live in `build_stt_and_turn_strategies()` in [`bot.py`](bot.py); switch with one env var.

## LLM

The LLM is built in [`llm.py`](llm.py). `OpenAILLMService` takes `base_url` + `api_key`,
so any OpenAI-compatible endpoint (e.g. Gemma via SGLang) works with env vars:

```bash
LLM_BASE_URL=https://<your-sglang-host>/v1
LLM_MODEL=google/gemma-4-26B-A4B-it
LLM_API_KEY=<token>
LLM_TEMPERATURE=0.6   # optional
```

Leave `LLM_BASE_URL` unset to use direct OpenAI (`OPENAI_MODEL`, default `gpt-4o`).

## Configuration

All knobs are environment variables — see [`.env.example`](.env.example) for the full list.
