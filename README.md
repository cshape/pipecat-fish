# pipecat-fish

A minimal [Pipecat](https://docs.pipecat.ai) voice agent that speaks with
[Fish Audio](https://fish.audio) TTS. It's the **same service stack as a
[LiveKit Agents](https://docs.livekit.io/agents/) voice bot**, rebuilt on Pipecat
— so you can run Fish on Pipecat and sub in a self-hosted LLM the same way.

You talk to it in the browser; it transcribes, runs an LLM, and replies in a
Fish voice.

## Stack

| Role | Service | Pipecat extra |
|------|---------|---------------|
| STT  | **Deepgram Flux** (default) or AssemblyAI | `pipecat-ai[deepgram]` / `[assemblyai]` |
| LLM  | OpenAI-compatible — `gpt-*`, **or self-hosted Gemma** via `LLM_BASE_URL` | `pipecat-ai[openai]` |
| TTS  | Fish Audio `s2.1-pro`, PCM @ 24 kHz, low latency | `pipecat-ai[fish]` |
| VAD  | Silero (barge-in / interruptions) | `pipecat-ai[silero]` |
| Transport | Small WebRTC (local) / Daily | `pipecat-ai[webrtc,daily,runner]` |

## Quick start

```bash
cp .env.example .env.local        # fill in DEEPGRAM_API_KEY, FISH_API_KEY, and either OPENAI_API_KEY or the LLM_* (Gemma) vars
uv sync
uv run bot.py
```

Then open <http://localhost:7860>, allow the mic, and talk.

## Turn-taking

How the agent decides you've finished speaking depends on `STT_PROVIDER`:

- **`deepgram`** (default) — Deepgram **Flux** does end-of-turn detection itself,
  server-side on the same audio stream. There's no separate turn model in the
  loop, which makes it the snappier path. Tune with `DEEPGRAM_EOT_THRESHOLD`
  (0.5–0.9; lower = faster, more false ends) and `DEEPGRAM_EOT_TIMEOUT_MS` (max
  silence before forcing the turn). Silero VAD still runs, but only for barge-in.
- **`assemblyai`** — Silero VAD gates speech and Pipecat's bundled **smart-turn-v3**
  ONNX model decides end-of-turn (the analog to LiveKit's turn detector). Tune the
  worst-case wait with `SMART_TURN_STOP_SECS` (default 0.6 s).

Both paths are in [`build_stt_and_turn_strategies()`](bot.py); switch with one env var.

## Subbing in Gemma (or any OpenAI-compatible LLM)

The LLM is built in [`llm.py`](llm.py). Pipecat's `OpenAILLMService` takes
`base_url` + `api_key`, so pointing it at a self-hosted model (e.g. Gemma served
OpenAI-compatible via SGLang) needs no fork — just env vars:

```bash
LLM_BASE_URL=https://<your-sglang-host>/v1
LLM_MODEL=google/gemma-4-26B-A4B-it
LLM_API_KEY=<token>
LLM_TEMPERATURE=0.6   # optional
```

Leave `LLM_BASE_URL` unset to fall back to direct OpenAI (`OPENAI_MODEL`, default `gpt-4o`).

## Configuration

All knobs are environment variables — see [`.env.example`](.env.example) for the
full list (STT provider + keys, LLM endpoint, Fish voice/format, and per-provider
turn-detection tuning).

## Notes

- Fish runs as **PCM @ 24 kHz** (not WAV) to avoid the container-decode path that
  adds an audible first-word crackle over WebRTC.
- This is the basic preset-voice loop. Extras like voice cloning, expressive
  register switching, and Flux **eager end-of-turn** (speculative early
  generation) are intentionally left out — build them on top of this skeleton.
- Docs: [Fish TTS](https://docs.pipecat.ai/api-reference/server/services/tts/fish) ·
  [Deepgram Flux](https://developers.deepgram.com/docs/flux/configuration) ·
  [Pipecat quickstart](https://docs.pipecat.ai/pipecat/get-started/quickstart)
