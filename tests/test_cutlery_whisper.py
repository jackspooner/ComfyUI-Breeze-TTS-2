import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_bridge():
    registry = ModuleType("nodes")
    registry.NODE_CLASS_MAPPINGS = {}
    loader = ModuleType("breeze_test.loader")
    for name in ("ATTENTION_OPTIONS", "DECODE_MODE_OPTIONS", "DEVICE_OPTIONS", "DTYPE_OPTIONS"):
        setattr(loader, name, ["auto"])
    loader.REPO_CHOICES = {}
    package = ModuleType("breeze_test")
    package.__path__ = [str(ROOT)]
    modules = {"nodes": registry, "breeze_test": package, "breeze_test.loader": loader,
               "breeze_test.native": ModuleType("breeze_test.native"),
               "breeze_test.runtime": SimpleNamespace(SAMPLE_RATE=24000)}
    with patch.dict(sys.modules, modules):
        loaded = []
        for name in ("whisper", "nodes"):
            spec = importlib.util.spec_from_file_location(f"breeze_test.{name}", ROOT / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            setattr(package, name, module)
            loaded.append(module)
    return registry, loaded[0], loaded[1]


class CutleryWhisperTests(unittest.TestCase):
    def setUp(self):
        self.registry, self.whisper, self.nodes = load_bridge()
        self.audio = {"waveform": torch.zeros(1, 2, 24000), "sample_rate": 24000}
        self.context = {"owner": "Cutlery"}
        self.loader = Mock()
        self.loader.load.return_value = (self.context,)
        self.transcriber = Mock()
        self.transcriber.transcribe.return_value = ("  Hello, world!  ", "lyrics", "{}", "timed")
        self.registry.NODE_CLASS_MAPPINGS.update({
            "CutleryFasterWhisperLoader": Mock(return_value=self.loader),
            "CutleryFasterWhisperTranscribeLyrics": Mock(return_value=self.transcriber),
        })

    def test_helper_delegates_controls_and_preserves_original_audio(self):
        result = self.whisper.BreezeWhisperTranscribe().transcribe(
            self.audio, "whisper-large-v3-turbo", "bf16", "chinese", "translate", 15, True)
        self.assertIs(result[0], self.audio)
        self.assertEqual(result[1], "Hello, world!")
        self.loader.load.assert_called_once_with(model_name="large-v3-turbo", device="auto", compute_type="bfloat16")
        self.transcriber.transcribe.assert_called_once_with(
            asr_model=self.context, audio=self.audio, language="zh", task="translate",
            chunk_length_s=15, strip_punctuation=False)
        self.assertFalse(hasattr(self.whisper, "_PIPELINE_CACHE"))

    def test_all_saved_model_and_precision_choices_map_to_cutlery(self):
        for model in self.whisper.POPULAR_WHISPER_MODELS:
            for dtype, compute in (("auto", "auto"), ("fp32", "float32"), ("bf16", "bfloat16")):
                with self.subTest(model=model, dtype=dtype):
                    self.whisper.BreezeWhisperTranscribe().transcribe(self.audio, model, dtype, "auto", "transcribe", 0, False)
                    self.assertEqual(self.loader.load.call_args.kwargs["model_name"], model.removeprefix("whisper-"))
                    self.assertEqual(self.loader.load.call_args.kwargs["compute_type"], compute)
                    self.assertEqual(self.transcriber.transcribe.call_args.kwargs["language"], "")

    def test_missing_cutlery_and_missing_weights_fail_without_fallback(self):
        self.registry.NODE_CLASS_MAPPINGS.clear()
        with self.assertRaisesRegex(RuntimeError, "requires Cutlery-Nodes"):
            self.whisper.BreezeWhisperTranscribe().transcribe(self.audio, "whisper-tiny", "auto", "auto", "transcribe", 30, True)
        self.registry.NODE_CLASS_MAPPINGS.update({
            "CutleryFasterWhisperLoader": Mock(return_value=self.loader),
            "CutleryFasterWhisperTranscribeLyrics": Mock(return_value=self.transcriber),
        })
        self.transcriber.transcribe.side_effect = FileNotFoundError("Install the selected CTranslate2 model with Download Manager 2")
        with self.assertRaisesRegex(FileNotFoundError, "Download Manager 2"):
            self.whisper.BreezeWhisperTranscribe().transcribe(self.audio, "whisper-tiny", "auto", "auto", "transcribe", 30, True)

    def test_speaker_auto_transcription_preserves_bundle_outputs_and_ui(self):
        result = self.nodes.BreezeTTS2Speaker().prepare("Ada", "none", "", True, "", 1, reference_audio=self.audio)
        speaker, preview, transcript = result["result"]
        self.assertIs(preview, self.audio)
        self.assertIs(speaker["ref_audio"], self.audio)
        self.assertEqual(speaker["ref_text"], "Hello, world!")
        self.assertEqual(transcript, "Hello, world!")
        self.assertIn("auto-transcript", result["ui"]["text"][0])
        self.loader.load.assert_called_once_with(model_name="large-v3-turbo", device="auto", compute_type="auto")

    def test_manual_and_instruction_only_speakers_do_not_load_whisper(self):
        speaker = self.nodes.BreezeTTS2Speaker()
        manual = speaker.prepare("Ada", "none", " Exact words. ", True, "", 1, reference_audio=self.audio)
        self.assertEqual(manual["result"][2], "Exact words.")
        designed = speaker.prepare("Bob", "none", "", True, "Warm voice", 4)
        self.assertEqual(designed["result"][2], "")
        self.assertEqual(designed["result"][1]["sample_rate"], 24000)
        self.loader.load.assert_not_called()
        with self.assertRaisesRegex(ValueError, "no transcript"):
            speaker.prepare("Ada", "none", "", False, "", 1, reference_audio=self.audio)

    def test_file_speaker_and_empty_transcript_preserve_existing_behavior(self):
        folder_paths = SimpleNamespace(get_annotated_filepath=Mock(return_value="/fixture/reference.wav"))
        with patch.dict(sys.modules, {"folder_paths": folder_paths}), patch.object(self.nodes, "_load_audio_file", return_value=self.audio):
            result = self.nodes.BreezeTTS2Speaker().prepare("Ada", "reference.wav", "", True, "", 1)
        self.assertIs(result["result"][1], self.audio)
        self.transcriber.transcribe.return_value = ("  ", "", "{}", "")
        with self.assertRaisesRegex(ValueError, "no transcript"):
            self.nodes.BreezeTTS2Speaker().prepare("Ada", "none", "", True, "", 1, reference_audio=self.audio)

    def test_saved_input_order_and_return_sockets_remain_compatible(self):
        helper = self.whisper.BreezeWhisperTranscribe
        self.assertEqual(list(helper.INPUT_TYPES()["required"]),
                         ["audio", "model", "dtype", "language", "task", "chunk_length_s", "download_if_missing"])
        self.assertEqual(helper.RETURN_TYPES, ("AUDIO", "STRING"))
        self.assertEqual(self.nodes.BreezeTTS2Speaker.RETURN_TYPES, ("BREEZE_SPEAKER", "AUDIO", "STRING"))


if __name__ == "__main__":
    unittest.main()
