# CLAUDE.md

Guidance for working in this repo.

## What this is

A minimal [Pipecat](https://docs.pipecat.ai) voice agent: browser ↔ STT → LLM →
Fish Audio TTS. Two files hold all the logic:

- [`bot.py`](bot.py) — the pipeline, service construction, turn-taking, and the
  runner entry point.
- [`llm.py`](llm.py) — the LLM seam (`build_llm`): OpenAI-compatible, points at
  self-hosted Gemma when `LLM_BASE_URL` is set, else direct OpenAI.

There is no separate frontend — `uv run bot.py` serves Pipecat's prebuilt WebRTC
client at <http://localhost:7860>.

## Architecture notes (the non-obvious parts)

- **STT is pluggable via `STT_PROVIDER`** (`deepgram` default | `assemblyai`), in
  `build_stt_and_turn_strategies()`. The provider choice is coupled to turn-taking:
  - **Deepgram Flux** does end-of-turn detection itself and emits the turn events,
    so it uses `ExternalUserTurnStrategies()` — no separate turn model.
  - **AssemblyAI** does not, so it's paired with the bundled `LocalSmartTurnAnalyzerV3`
    (ONNX, ships with pipecat, no download) as the stop strategy.
  - Silero VAD runs in both cases, but only for barge-in/interruption — it is NOT
    the end-of-turn decider.
- **Turn-taking is the latency lever**, not TTFB. If the agent feels sluggish,
  it's almost always the end-of-turn wait: `DEEPGRAM_EOT_THRESHOLD` /
  `DEEPGRAM_EOT_TIMEOUT_MS` for Flux, `SMART_TURN_STOP_SECS` for smart-turn.
- **Fish runs PCM @ 24 kHz on purpose** — the WAV container path adds an audible
  first-word crackle over WebRTC. Don't switch `FISH_OUTPUT_FORMAT` to `wav`.
- **Pipecat auto-enables RTVI** (`PipelineTask(enable_rtvi=True)`); that's what
  drives the prebuilt client's event panel. The panel logs each spoken sentence on
  two channels (`spoken: false` = LLM text, `spoken: true` = TTS) — this is a
  debug-client display artifact, not duplicated audio. `bot-output` is all-or-nothing,
  so you can't keep only the spoken variant in that client.
- **Pin caveat**: code against the *installed* pipecat version's signatures, not
  the public docs — the docs sometimes run ahead of the released package (e.g.
  `OpenAILLMService` takes `base_url` via the base class, not a named kwarg in the
  released version). Verify with `inspect.signature(...)` when unsure.

## Conventions

- Python ≥ 3.10, managed with `uv` (`uv sync`, `uv run`). `uv.lock` is committed.
- All configuration is environment variables; defaults live in `bot.py` and the
  documented list is in `.env.example`. Keep those two in sync when adding a knob.
- Lint/format: `uv run ruff check` / `uv run ruff format`.

## Secrets

- Real keys live in `.env.local`, which is gitignored. **Never commit it.**
- `.env.example` must stay placeholder-only.

## Common tasks

- Run the agent: `uv run bot.py`
- Switch STT: set `STT_PROVIDER=assemblyai` (or `deepgram`) in `.env.local`
- Use Gemma: set `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY` in `.env.local`

## Possible next steps (not yet built)

- Flux **eager end-of-turn** (`eager_eot_threshold`) for speculative early LLM
  generation — needs handling `EagerEndOfTurn` / `TurnResumed` and pairs with
  preemptive generation.
- Word-level **timestamps**: Fish offers them via the SSE `/v1/tts/stream/with-timestamp`
  endpoint, but pipecat's `FishAudioTTSService` uses the WebSocket `/v1/tts/live`
  endpoint and neither requests nor parses alignment data — would need a custom service.
