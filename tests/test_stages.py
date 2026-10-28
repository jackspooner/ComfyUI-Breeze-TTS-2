import contextlib
import ast
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Params:
    max_new_tokens: int
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    depth_temperature: float
    depth_top_k: int
    depth_top_p: float


class StageTests(unittest.TestCase):
    def setUp(self):
        package = ModuleType("breeze_stage_test")
        package.__path__ = [str(ROOT)]
        self.calls = []
        def build(*args, **kwargs):
            self.calls.append("prepare")
            self.assertEqual(kwargs["negative_segments"], ["negative"])
            return torch.rand(2, 4, 3), torch.ones(2, 4), torch.tensor([4, 3]), 4
        def generate(model, **kwargs):
            self.calls.append("generate")
            self.assertEqual(kwargs["params"].max_new_tokens, 100)
            self.assertTrue(torch.equal(kwargs["base_positions"], torch.tensor([4, 3])))
            return kwargs["inputs_embeds"].flatten() + torch.rand(24)
        self.runtime = SimpleNamespace(fix_seed=torch.manual_seed, MAX_SEQ_LEN=2048,
            GenerationParams=Params, design_segments=lambda *args: ["positive"],
            design_negative_segments=lambda text: ["negative"], build_generation_batch=build,
            generate_codes=generate, decode_codes=lambda codec, codes: codes,
            tensor_audio_to_comfy=lambda wav: {"waveform": wav, "sample_rate": 24000})
        self.loader = SimpleNamespace(resume_bundle_to_device=Mock())
        self.native = SimpleNamespace(attention_runtime=lambda attention: contextlib.nullcontext())
        modules = {"breeze_stage_test": package, "breeze_stage_test.loader": self.loader,
                   "breeze_stage_test.native": self.native, "breeze_stage_test.runtime": self.runtime}
        with patch.dict(sys.modules, modules):
            spec = importlib.util.spec_from_file_location("breeze_stage_test.stages", ROOT / "stages.py")
            self.stages = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.stages)
        self.bundle = SimpleNamespace(model=object(), codec=object(), tokenizer=object(), model_dir=Path("model"),
            weights_name="weights", device=torch.device("cpu"), dtype_name="fp32", attention="eager",
            decode_mode="eager", quantized=False)
        self.controls = dict(text="speech", instruction="warm", cfg_scale=4, max_new_tokens=100,
            temperature=1, top_k=20, top_p=.8, repetition_penalty=1.1, depth_temperature=1,
            depth_top_k=20, depth_top_p=.8)

    def test_interleaved_children_keep_rng_masks_and_exact_outputs(self):
        prepared = [self.stages.prepare(self.bundle, seed=seed, **self.controls) for seed in (41, 42)]
        generated = [self.stages.BreezeTTS2GenerateCodes().run(self.bundle, state)[0] for state in prepared]
        for seed, state in zip((41, 42), generated):
            audio = self.stages.BreezeTTS2DecodeCodes().run(self.bundle, state)[0]
            torch.manual_seed(seed)
            expected = torch.rand(2, 4, 3).flatten() + torch.rand(24)
            self.assertTrue(torch.equal(audio["waveform"], expected))
            self.assertEqual(audio["sample_rate"], 24000)
        self.assertEqual(self.calls, ["prepare", "prepare", "generate", "generate"])
        self.assertEqual(self.loader.resume_bundle_to_device.call_count, 6)

    def test_identity_phase_limits_and_invalid_audio_fail_closed(self):
        state = self.stages.prepare(self.bundle, seed=41, **self.controls)
        self.bundle.quantized = True
        with self.assertRaisesRegex(ValueError, "BREEZE_STAGE_IDENTITY"):
            self.stages.BreezeTTS2GenerateCodes().run(self.bundle, state)
        self.bundle.quantized = False
        with self.assertRaisesRegex(ValueError, "BREEZE_STAGE_INVALID"):
            self.stages.BreezeTTS2DecodeCodes().run(self.bundle, state)
        with patch.object(self.stages, "MAX_STATE_BYTES", 1), self.assertRaisesRegex(ValueError, "BREEZE_STAGE_LIMIT"):
            self.stages.prepare(self.bundle, seed=41, **self.controls)
        state = self.stages.BreezeTTS2GenerateCodes().run(self.bundle, state)[0]
        self.runtime.decode_codes = lambda *args: torch.tensor([float("nan")])
        with self.assertRaises(RuntimeError):
            self.stages.BreezeTTS2DecodeCodes().run(self.bundle, state)

    def test_prompt_frame_budget_is_preserved(self):
        with self.assertRaisesRegex(ValueError, "audio frames"):
            self.stages.prepare(self.bundle, seed=41, **{**self.controls, "max_new_tokens": 63})

    def test_changed_artifact_fingerprint_invalidates_retained_state(self):
        self.bundle.artifact_identity = "before"
        state = self.stages.prepare(self.bundle, seed=41, **self.controls)
        self.bundle.artifact_identity = "after"
        with self.assertRaisesRegex(ValueError, "BREEZE_STAGE_IDENTITY"):
            self.stages.BreezeTTS2GenerateCodes().run(self.bundle, state)

    def test_loader_reloads_changed_digest_even_with_identical_path_and_mtime(self):
        # Execute the actual owner load function with model construction stopped
        # immediately after the reuse decision; no weights or GPU are involved.
        tree = ast.parse((ROOT / "loader.py").read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "load_breeze_bundle")
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
        class ReloadRequired(Exception):
            pass
        with tempfile.TemporaryDirectory() as directory:
            weight = Path(directory) / "weights"
            weight.touch()
            unload, resume = Mock(), Mock()
            namespace = {"_ACTIVE_BUNDLE": self.bundle,
                "_ACTIVE_LOAD_KEY": (directory, "weights", weight.stat().st_mtime_ns, "cpu", "fp32", "eager", "eager", "before"),
                "DECODE_MODE_OPTIONS": ["eager"], "resolve_model_dir": lambda *args: (Path(directory), "weights"),
                "weights_path": lambda *args: weight, "resolve_device": lambda *args: "cpu",
                "resolve_dtype_mode": lambda *args: "fp32", "resolve_attention": lambda *args: "eager",
                "resume_bundle_to_device": resume, "unload_breeze_bundle": unload,
                "int8": SimpleNamespace(scan_checkpoint_quantization=Mock(side_effect=ReloadRequired))}
            exec(compile(ast.fix_missing_locations(module), "loader.py", "exec"), namespace)
            load = namespace["load_breeze_bundle"]
            self.assertIs(load("model", "fp32", "cpu", "eager", False, artifact_identity="before"), self.bundle)
            unload.assert_not_called()
            with patch.dict(sys.modules, {"transformers": SimpleNamespace(AutoTokenizer=object)}), self.assertRaises(ReloadRequired):
                load("model", "fp32", "cpu", "eager", False, artifact_identity="after")
            unload.assert_called_once_with(self.bundle, reason="load settings changed")
