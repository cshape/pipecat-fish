#
# Pipecat + Fish Audio voice agent.
#
# Mirrors the service stack of ../livekit-demo/fish on Pipecat instead of
# LiveKit Agents:
#
#   STT  - Deepgram Flux (default) or AssemblyAI   (set STT_PROVIDER)
#   LLM  - OpenAI-compatible     (gpt-* by default, or self-hosted Gemma via LLM_BASE_URL — see llm.py)
#   TTS  - Fish Audio s2.1-pro   (PCM @ 24kHz, low latency)
#   VAD  - Silero (barge-in)
#
# Turn-taking depends on STT_PROVIDER:
#   deepgram  -> Flux's native end-of-turn drives turns (ExternalUserTurnStrategies)
#   assemblyai-> Silero VAD + bundled smart-turn-v3 model decides end-of-turn
#
# Run it (opens a local WebRTC client at http://localhost:7860):
#
#     uv run bot.py
#

import os

from dotenv import load_dotenv
from loguru import logger

print("🚀 Starting Pipecat + Fish bot...")
print("⏳ Loading models and imports (first run takes ~20s)\n")

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.assemblyai.models import AssemblyAIConnectionParams
from pipecat.services.assemblyai.stt import AssemblyAISTTService
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.services.fish.tts import FishAudioTTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import (
    ExternalUserTurnStrategies,
    UserTurnStrategies,
)

from llm import build_llm

load_dotenv(".env.local")
load_dotenv()  # fall back to .env if .env.local is absent

# --- Config -----------------------------------------------------------------

SYSTEM_INSTRUCTION = (
    "You are a warm, curious voice companion. Keep replies to one or two short, "
    "conversational sentences — this is spoken aloud, so no lists, markdown, or "
    "headings. Open warmly, then turn it back to the user with a genuine question."
)

GREETING = "Say hello warmly and briefly introduce yourself in one short sentence, then ask how I'm doing."

# Default Fish voice (Maren, American F) — same preset the LiveKit demo opens in.
# Override per deployment with FISH_VOICE_ID (a Fish reference/model id).
DEFAULT_FISH_VOICE_ID = "0e24ff9936d34df4bddce26398cf1311"

# Fish s2.1-pro over PCM @ 24kHz, low latency — matches livekit-demo/fish/src/tts_factory.py.
# PCM (over WAV) avoids the container-decode path that adds a first-word crackle.
FISH_MODEL = os.getenv("FISH_MODEL", "s2.1-pro")
FISH_OUTPUT_FORMAT = os.getenv("FISH_OUTPUT_FORMAT", "pcm")
FISH_SAMPLE_RATE = int(os.getenv("FISH_SAMPLE_RATE", "24000"))
FISH_LATENCY = os.getenv("FISH_LATENCY", "low")

# --- STT provider ------------------------------------------------------------
# "deepgram" (Flux) | "assemblyai". Default deepgram — Flux has integrated,
# server-side end-of-turn detection (no separate turn model), which is the
# faster path for a conversational agent.
STT_PROVIDER = os.getenv("STT_PROVIDER", "deepgram").lower()

# --- Deepgram Flux turn detection (snappiness) ------------------------------
# Flux decides end-of-turn itself. eot_threshold = confidence to end the turn
# (0.5–0.9; lower = snappier, more false ends). eot_timeout_ms = max silence
# before forcing the turn regardless of confidence (default 5000; lower = snappier
# on trailing pauses). Eager EOT (speculative early generation) is left OFF — it
# needs explicit handling of EagerEndOfTurn/TurnResumed cancellation.
DEEPGRAM_FLUX_MODEL = os.getenv("DEEPGRAM_FLUX_MODEL", "flux-general-en")
DEEPGRAM_EOT_THRESHOLD = float(os.getenv("DEEPGRAM_EOT_THRESHOLD", "0.7"))
DEEPGRAM_EOT_TIMEOUT_MS = int(os.getenv("DEEPGRAM_EOT_TIMEOUT_MS", "3000"))

# --- AssemblyAI + smart-turn (used only when STT_PROVIDER=assemblyai) --------
# Silero VAD *gates* speech, the bundled smart-turn-v3 ONNX model decides
# end-of-turn. STOP_SECS is the worst-case wait when smart-turn keeps predicting
# "incomplete"; when confident it ends in ~200ms. 0.6s tolerates a brief pause.
SMART_TURN_STOP_SECS = float(os.getenv("SMART_TURN_STOP_SECS", "0.6"))
# AssemblyAI (u3-rt-pro) returns finals after this much silence so smart-turn has
# the transcript ready the instant it calls the turn. 100ms = snappy.
ASSEMBLYAI_MIN_TURN_SILENCE_MS = int(os.getenv("ASSEMBLYAI_MIN_TURN_SILENCE_MS", "100"))


def build_stt_and_turn_strategies():
    """Return (stt_service, user_turn_strategies) for the configured provider.

    Deepgram Flux owns turn detection (so we use ExternalUserTurnStrategies and
    let Flux's EndOfTurn drive the turn). AssemblyAI does not, so we pair it with
    the local smart-turn-v3 analyzer as the stop strategy.
    """
    if STT_PROVIDER == "deepgram":
        stt = DeepgramFluxSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            model=DEEPGRAM_FLUX_MODEL,
            settings=DeepgramFluxSTTService.Settings(
                eot_threshold=DEEPGRAM_EOT_THRESHOLD,
                eot_timeout_ms=DEEPGRAM_EOT_TIMEOUT_MS,
                min_confidence=0.3,
            ),
        )
        # Flux emits the user turn start/stop itself; the aggregator just follows it.
        return stt, ExternalUserTurnStrategies()

    if STT_PROVIDER == "assemblyai":
        stt = AssemblyAISTTService(
            api_key=os.getenv("ASSEMBLYAI_API_KEY"),
            vad_force_turn_endpoint=True,
            connection_params=AssemblyAIConnectionParams(
                min_turn_silence=ASSEMBLYAI_MIN_TURN_SILENCE_MS,
            ),
        )
        turn_strategies = UserTurnStrategies(
            stop=[
                TurnAnalyzerUserTurnStopStrategy(
                    turn_analyzer=LocalSmartTurnAnalyzerV3(
                        params=SmartTurnParams(stop_secs=SMART_TURN_STOP_SECS),
                    ),
                ),
            ],
        )
        return stt, turn_strategies

    raise ValueError(f"unknown STT_PROVIDER={STT_PROVIDER!r} (use deepgram | assemblyai)")


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot")

    stt, user_turn_strategies = build_stt_and_turn_strategies()
    logger.info(f"STT provider = {STT_PROVIDER}")

    tts = FishAudioTTSService(
        api_key=os.getenv("FISH_API_KEY"),
        output_format=FISH_OUTPUT_FORMAT,
        sample_rate=FISH_SAMPLE_RATE,
        settings=FishAudioTTSService.Settings(
            voice=os.getenv("FISH_VOICE_ID", DEFAULT_FISH_VOICE_ID),
            model=FISH_MODEL,
            latency=FISH_LATENCY,
        ),
    )

    # Conversation LLM. Set LLM_BASE_URL / LLM_MODEL / LLM_API_KEY to point at
    # self-hosted Gemma; otherwise direct OpenAI. See llm.py.
    llm = build_llm(system_instruction=SYSTEM_INSTRUCTION)

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            # VAD still runs for barge-in/interruption detection. The *turn-stop*
            # decision comes from user_turn_strategies (Flux EOT or smart-turn).
            vad_analyzer=SileroVADAnalyzer(),
            user_turn_strategies=user_turn_strategies,
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Mic in
            stt,  # Deepgram Flux / AssemblyAI
            user_aggregator,  # User turn -> context
            llm,  # OpenAI-compatible (Gemma-capable)
            tts,  # Fish Audio
            transport.output(),  # Speaker out
            assistant_aggregator,  # Bot turn -> context
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # SmallWebRTC can fire on_client_connected more than once (renegotiation),
    # which would queue a fresh greeting each time — guard so we greet exactly once.
    greeted = False

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        nonlocal greeted
        if greeted:
            logger.info("Client (re)connected — greeting already sent, skipping")
            return
        greeted = True
        logger.info("Client connected")
        context.add_message({"role": "developer", "content": GREETING})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Entry point for the Pipecat bot runner."""
    transport_params = {
        "daily": lambda: DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    }

    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
