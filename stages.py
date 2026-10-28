"""Independent voice-design phases with safe CPU tensor boundaries."""
from dataclasses import asdict

import torch

from . import loader, native, runtime

VERSION = 1
MAX_STATE_BYTES = 512 * 1024**2


def identity(bundle):
    return {"model_dir": str(bundle.model_dir), "weights_name": bundle.weights_name,
            "device": str(bundle.device), "dtype_name": bundle.dtype_name,
            "attention": bundle.attention, "decode_mode": bundle.decode_mode,
            "quantized": bundle.quantized}


def cpu_tensors(tensors):
    size = sum(value.numel() * value.element_size() for value in tensors.values())
    if size > MAX_STATE_BYTES:
        raise ValueError("BREEZE_STAGE_LIMIT: tensor state exceeds 512 MiB")
    return {key: value.detach().cpu().clone() for key, value in tensors.items()}


def rng_state(device):
    state = {"cpu": torch.get_rng_state()}
    if torch.device(device).type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng(state, device):
    torch.set_rng_state(state["cpu"])
    if torch.device(device).type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def validate(state, bundle, phase):
    if not isinstance(state, dict) or state.get("version") != VERSION or state.get("phase") != phase:
        raise ValueError("BREEZE_STAGE_INVALID: retained version or phase mismatch")
    if state.get("identity") != identity(bundle):
        raise ValueError("BREEZE_STAGE_IDENTITY: model, precision, attention or device changed")


def prepare(bundle, *, text, instruction, cfg_scale, seed, max_new_tokens, **controls):
    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")
    loader.resume_bundle_to_device(bundle)
    runtime.fix_seed(int(seed))
    with torch.inference_mode():
        embeds, mask, positions, prefill = runtime.build_generation_batch(
            bundle.model, bundle.tokenizer, cond_segments=runtime.design_segments(text, instruction),
            negative_segments=runtime.design_negative_segments(text) if cfg_scale != 1.0 else None,
            ref_codes=None, cfg_scale=float(cfg_scale), device=bundle.device)
    frames = min(int(max_new_tokens), runtime.MAX_SEQ_LEN - 1 - prefill)
    if frames < 64:
        raise ValueError(f"The prompt alone is {prefill} tokens, leaving only {frames} audio frames.")
    params = runtime.GenerationParams(max_new_tokens=frames, **{
        name: (int(value) if name in ("top_k", "depth_top_k") else float(value))
        for name, value in controls.items()})
    return {"version": VERSION, "phase": "prepared", "identity": identity(bundle),
            "cfg_scale": float(cfg_scale), "prefill_len": prefill, "params": asdict(params),
            "rng": rng_state(bundle.device), "tensors": cpu_tensors({"inputs_embeds": embeds,
            "attention_mask": mask, "base_positions": positions})}


class BreezeTTS2PrepareVoiceDesign:
    @classmethod
    def INPUT_TYPES(cls):
        from .nodes import BreezeTTS2VoiceDesign
        return BreezeTTS2VoiceDesign.INPUT_TYPES()

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("state",)
    FUNCTION = "run"
    CATEGORY = "Breeze TTS 2/stages"
    DESCRIPTION = "Prepare one voice-design request's embeddings, CFG masks and positions for retained stages."

    def run(self, breeze_model, **controls):
        return (prepare(breeze_model, **controls),)


class BreezeTTS2GenerateCodes:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"breeze_model": ("BREEZE_TTS2_MODEL",), "state": ("LATENT",)}}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("state",)
    FUNCTION = "run"
    CATEGORY = "Breeze TTS 2/stages"
    DESCRIPTION = "Generate one child's codec codes using the existing backbone/depth runtime."

    def run(self, breeze_model, state):
        validate(state, breeze_model, "prepared")
        loader.resume_bundle_to_device(breeze_model)
        restore_rng(state["rng"], breeze_model.device)
        with torch.inference_mode(), native.attention_runtime(breeze_model.attention):
            codes = runtime.generate_codes(breeze_model.model,
                **{name: value.to(breeze_model.device) for name, value in state["tensors"].items()},
                prefill_len=state["prefill_len"], cfg_scale=state["cfg_scale"],
                params=runtime.GenerationParams(**state["params"]), decode_mode=breeze_model.decode_mode)
        return ({"version": VERSION, "phase": "generated", "identity": state["identity"],
                 "rng": rng_state(breeze_model.device), "tensors": cpu_tensors({"codes": codes})},)


class BreezeTTS2DecodeCodes(BreezeTTS2GenerateCodes):
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    DESCRIPTION = "Decode a retained child's Breeze codec codes into 24 kHz mono audio."

    def run(self, breeze_model, state):
        validate(state, breeze_model, "generated")
        loader.resume_bundle_to_device(breeze_model)
        restore_rng(state["rng"], breeze_model.device)
        wav = runtime.decode_codes(breeze_model.codec, state["tensors"]["codes"])
        if not bool(torch.isfinite(wav).all()) or wav.numel() == 0:
            raise RuntimeError("Breeze TTS 2 produced empty or non-finite audio.")
        return (runtime.tensor_audio_to_comfy(wav),)


NODE_CLASS_MAPPINGS = {node.__name__: node for node in
    (BreezeTTS2PrepareVoiceDesign, BreezeTTS2GenerateCodes, BreezeTTS2DecodeCodes)}
NODE_DISPLAY_NAME_MAPPINGS = {"BreezeTTS2PrepareVoiceDesign": "Breeze TTS 2 Prepare Voice Design",
    "BreezeTTS2GenerateCodes": "Breeze TTS 2 Generate Codes", "BreezeTTS2DecodeCodes": "Breeze TTS 2 Decode Codes"}
