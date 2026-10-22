"""Whisper ASR node for producing reference transcripts for Breeze TTS 2 voice cloning."""

from __future__ import annotations

import nodes as comfy_nodes

CATEGORY = "Breeze TTS 2"

POPULAR_WHISPER_MODELS = {
    "whisper-large-v3-turbo": "large-v3-turbo",
    "whisper-large-v3": "large-v3",
    "whisper-medium": "medium",
    "whisper-small": "small",
    "whisper-base": "base",
    "whisper-tiny": "tiny",
}

LANGUAGE_CHOICES = [
    "auto", "english", "chinese", "german", "spanish", "russian", "korean",
    "french", "japanese", "portuguese", "turkish", "polish", "catalan",
]

DTYPE_OPTIONS = ["auto", "bf16", "fp32"]
LANGUAGE_CODES = dict(zip(LANGUAGE_CHOICES, ("", "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca")))


def whisper_model_choices() -> list[str]:
    return list(POPULAR_WHISPER_MODELS)


class BreezeWhisperTranscribe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Audio to transcribe (e.g. the reference clip used for cloning)."}),
                "model": (
                    whisper_model_choices(),
                    {"default": "whisper-large-v3-turbo", "tooltip": "Preinstalled Cutlery Faster-Whisper model used for transcription."},
                ),
                "dtype": (DTYPE_OPTIONS, {"default": "auto", "tooltip": "Compute dtype for Whisper."}),
                "language": (
                    LANGUAGE_CHOICES,
                    {"default": "auto", "tooltip": "Spoken language hint; auto detects it."},
                ),
                "task": (
                    ["transcribe", "translate"],
                    {"default": "transcribe", "tooltip": "transcribe keeps the source language; translate writes English."},
                ),
                "chunk_length_s": (
                    "INT",
                    {"default": 30, "min": 0, "max": 120, "tooltip": "Chunk length in seconds for long inputs (0 uses the engine default)."},
                ),
                "download_if_missing": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Retained for saved workflows. Whisper weights must be installed through Download Manager 2; neither value enables downloads."},
                ),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "transcript")
    FUNCTION = "transcribe"
    CATEGORY = CATEGORY
    DESCRIPTION = "Transcribe reference audio using Cutlery's managed Faster-Whisper nodes. Requires Cutlery-Nodes and preinstalled CTranslate2 weights."

    def transcribe(self, audio, model, dtype, language, task, chunk_length_s, download_if_missing):
        try:
            loader_class = comfy_nodes.NODE_CLASS_MAPPINGS["CutleryFasterWhisperLoader"]
            transcribe_class = comfy_nodes.NODE_CLASS_MAPPINGS["CutleryFasterWhisperTranscribeLyrics"]
        except KeyError as exc:
            raise RuntimeError("Breeze Whisper requires Cutlery-Nodes. Enable the package and restart idle ComfyUI before transcription.") from exc

        (asr_model,) = loader_class().load(
            model_name=POPULAR_WHISPER_MODELS[model],
            device="auto",
            compute_type={"auto": "auto", "bf16": "bfloat16", "fp32": "float32"}[dtype],
        )
        transcript, _lyrics, _segments, _timed = transcribe_class().transcribe(
            asr_model=asr_model,
            audio=audio,
            language=LANGUAGE_CODES[language],
            task=task,
            chunk_length_s=int(chunk_length_s),
            strip_punctuation=False,
        )
        return (audio, transcript.strip())


NODE_CLASS_MAPPINGS = {
    "BreezeWhisperTranscribe": BreezeWhisperTranscribe,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BreezeWhisperTranscribe": "Breeze TTS 2 Whisper Transcribe",
}
