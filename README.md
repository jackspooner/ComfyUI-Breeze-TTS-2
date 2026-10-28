# ComfyUI-Breeze-TTS-2

## Workspace2 fork

This fork derives from [Saganaki22/ComfyUI-Breeze-TTS-2](https://github.com/Saganaki22/ComfyUI-Breeze-TTS-2)
and its [upstream README](https://github.com/Saganaki22/ComfyUI-Breeze-TTS-2/blob/main/README.md).
Upstream provides bilingual voice cloning, voice design, voice direction,
multi-speaker dialogue, quantized checkpoints and ComfyUI model management.
The upstream documentation and licensing information are retained below.

This fork retains the organized local model path and delegates Whisper reference
transcription to the separately installed Cutlery-Nodes implementation. See the
managed Whisper and development sections below for requirements and limitations.
It also exposes retained voice-design stages using the original native runtime.

## Voice-design stages

`BreezeTTS2PrepareVoiceDesign` accepts the original `BreezeTTS2VoiceDesign`
inputs and returns `LATENT` state. Connect it to
`BreezeTTS2GenerateCodes.state`, then its state to
`BreezeTTS2DecodeCodes.state` for 24 kHz mono AUDIO. Generate and decode also
require `breeze_model` from the same loader settings; include the loader in
each native stage graph. Existing combined nodes remain compatible.

Preparation retains CPU embeddings, conditional/unconditional CFG masks, base
positions, actual prefill length, bounded frame cap, sampling controls and RNG
state. Generated state retains private codec codes and the decoder's starting
RNG. Model directory/weight name, selected device, precision, quantization,
attention and decode mode are checked at each consuming phase. Checkpoint paths
must stay immutable during the batch; the owner plan binds exact revisions and
reference/input hashes. The original loader/patchers retain ComfyUI admission
and eviction ownership. No model objects or live KV-cache handles cross stages.

`LATENT` is a safe tensor/data metadata carrier here, not diffusion samples.
Use the native stage engine for durable child identities, receipts, retention,
bounded groups and cancellation between units. It must schedule prepare units
before generation and generation before decode; plain graph JSON ordering does
not enforce these barriers. Export is capped at 512 MiB per state. The original
runtime continues checking interruption during token generation. Partial units
do not publish a successful stage receipt.

These nodes stage independent voice-design requests sequentially. CFG's paired
rows are still branches of one request and are not cross-request tensor batching.
Voice cloning, direction, dialogue and the managed Whisper bridge retain their
existing combined interfaces. CPU tests cover interleaved seeded requests,
masks/positions/frame limits, lifecycle delegation and invalid state/audio.
GPU memory, audio equivalence, listening quality and throughput need separate
model-backed acceptance; no speedup is claimed from CPU checks.

## Development

Use a prepared ComfyUI Python with Python >=3.10, PyTorch, Torchaudio, NumPy and
Faster-Whisper installed. In a fresh worktree, run
`COMFYUI_PYTHON=<python> CUTLERY_NODES_ROOT=<cutlery-checkout> python3 tools/setup_worktree.py`.
Setup only checks prerequisites; it installs nothing, loads no weights and starts
no services. Run `<python> -m unittest discover -s tests -v` for verification.
See [AGENTS.md](AGENTS.md) for prepared-runtime verification and live activation requirements.

## Managed Whisper

Install [Cutlery-Nodes](https://github.com/jackspooner/Cutlery-Nodes) with the
`CutleryFasterWhisperLoader` and `CutleryFasterWhisperTranscribeLyrics` nodes,
including optional `task` and `chunk_length_s` inputs. Both the Whisper helper
and Speaker's empty-transcript path delegate to these registered nodes, sharing
Cutlery's model reuse, GPU admission, audio resampling and idle cache reclamation.
The default remains `whisper-large-v3-turbo`, mapped to Cutlery's `large-v3-turbo`.

Provision CTranslate2 model files under the configured ComfyUI models root at
`audio/transcription/faster-whisper/large-v3-turbo` before execution. See
[Cutlery's managed Whisper guide](https://github.com/jackspooner/Cutlery-Nodes/blob/main/docs/managed-whisper.md)
for files, dependency setup, other model selections and recovery. Existing
Transformers weights under `audio_encoders` are not CTranslate2 models. Missing
nodes or weights fail explicitly; there is no fallback pipeline or automatic download.

Existing node IDs, input order, model/language choices and output sockets remain
compatible. The Whisper helper returns the original AUDIO and trimmed transcript;
Speaker retains its speaker/AUDIO/transcript tuple and UI text. Manual transcripts
and instruction-only speakers do not run Whisper. Translation targets English and
quality depends on the selected model; turbo is primarily a transcription model.
Chunk length is forwarded to Faster-Whisper (0 uses its default). BF16 maps to
native CTranslate2 BF16 on CUDA and FP32 on CPU. The saved `download_if_missing`
input remains accepted, but both values use preinstalled weights only.

After installing matching sources, restart only an idle ComfyUI and refresh the
frontend. The isolated tests do not prove real-model transcription quality or
GPU release; those require queued live verification with installed weights.

## Upstream documentation

**English** | [简体中文](README_ZH.md)

<img width="1483" height="1170" alt="Screenshot 2026-08-26 160830" src="https://github.com/user-attachments/assets/95f2f346-472e-45a5-bd3e-625007c6cc39" />



https://github.com/user-attachments/assets/d4226032-3337-450b-83c6-ab9c732172d6




[Breeze TTS 2](https://huggingface.co/BreezeBlue/Breeze-TTS-2) nodes for ComfyUI: **voice clone**,
**voice design**, and **voice direction** with bilingual (EN/ZH) speech, inline vocal events,
INT8 ConvRot quantized builds, and full ComfyUI / AIMDO dynamic-VRAM integration.

## Nodes (category: `Breeze TTS 2`)

| Node | Purpose |
| --- | --- |
| **Breeze TTS 2 Load Model** | Loads the checkpoint with `dtype` (auto/bf16/fp32), `device` (auto/cuda/cpu), `attention` (auto/eager/sdpa/flash_attention/sageattention), `decode_mode` (eager/cuda_graphs), and `download_if_missing`. |
| **Breeze TTS 2 Voice Clone** | Clones a speaker from clean reference audio + its exact transcript (CFG 1.0). |
| **Breeze TTS 2 Voice Design** | Creates a voice from a natural-language description, no reference audio (CFG 4). |
| **Breeze TTS 2 Voice Direction** | Clones a reference voice and steers tone, emotion, pace, and delivery with an instruction (CFG 4). `stitch_reference` can play the original reference clip before or after the generated speech in the output. |
| **Breeze TTS 2 Whisper Transcribe** | Whisper helper that produces the exact transcript cloning needs. |
| **Breeze TTS 2 Speaker** | One named cast member for Multi-Speaker. Pick a reference clip from the input folder (browse/upload dropdown, or wire audio in) to clone it, or select `none` and write an instruction to design the voice. An empty transcript is auto-transcribed with Whisper and shown on the node. |
| **Breeze TTS 2 Multi-Speaker** | Generates a whole dialogue script — plain `Name:` lines or pasted JSON — with up to 8 wired speakers, freely mixing cloned and designed voices. |

Vocal events work inline in the text: `(laugh)`, `(cough)`, `(clears throat)`, `(sigh)` in English;
`[笑]`, `[咳嗽]`, `[清嗓子]`, `[叹气]` in Chinese. Match the instruction language to the text language
for design/direction. Output is 24 kHz mono.

## Multi-speaker dialogue



https://github.com/user-attachments/assets/bf124113-f9ef-4a5d-a501-5935d8fd7b60



Wire up to 8 **Speaker** nodes into the **Multi-Speaker** node's `speaker_1…8` inputs, then write
(or paste) the script. Speaker matching is forgiving about case, spaces, and punctuation
(`Ali G` matches a speaker named `alig`), and every speaker name is validated before any
generation starts, so a bad script fails instantly instead of after burning GPU time.

**Option 1 — plain text.** One `Name: line` per line. A line without a name continues the
previous speaker; `[Name]:` and markdown `**Name:**` also work:

```
Ada: (sigh) Who steered my ship into the harbor wall this time?
Bob: Technically, captain, the harbor steered into us.
Ada: That is not how harbors work, Bob.
Bob: If I may, the tide charts were very clear about it.
```

**Option 2 — pasted JSON.** Ideal for LLM-written scripts. If the text starts with `[` or `{`
it is parsed as JSON:

```json
[
  {"speaker": "Ada", "text": "(sigh) Who steered my ship into the harbor wall this time?"},
  {"speaker": "Bob", "text": "Technically, captain, the harbor steered into us."},
  {"speaker": "Ada", "text": "That is not how harbors work, Bob."}
]
```

Also accepted: a single `{"speaker": ..., "text": ...}` object, a wrapper like
`{"script": [...]}`, and the key aliases `speaker`/`name`/`character`/`role` and
`text`/`line`/`content`/`message`. Inline vocal events work identically in both formats.
The `text` widget can be converted to an input (right-click) to wire a string from any
text-generating node.

Before inference the console prints a summary —

```
[BreezeTTS2] Multi-speaker script: 4 speaker(s), 16 turn(s), ~75s of speech total | cast: Ada (clone), Bob (clone), Carol (design), Dave (design)
```

— then one progress bar per turn (`[3/16] Bob (~6s)`). Reference clips are encoded once per
speaker, each speaker keeps a stable seed offset across turns (so designed voices stay
consistent), and `pause_between_speakers` controls the gap. Reference clips: keep them under
~20 s (hard max 60 s). To get an LLM to write scripts in this format, hand it
[SKILL.md](SKILL.md).

<img width="776" height="1153" alt="Screenshot 2026-08-28 035631" src="https://github.com/user-attachments/assets/be2ad864-bb9d-42b2-a474-ad22315a4dad" />


## Checkpoints

The loader downloads into `ComfyUI/models/breezetts2/` when `download_if_missing` is enabled:

| Choice | File | Notes |
| --- | --- | --- |
| int8 hybrid **(default)** | `Breeze-TTS-2-int8-hybrid.safetensors` | Backbone + text encoder INT8 ConvRot, depth decoder bf16. bf16-level speed at −27 % VRAM. |
| bf16 combined | `Breeze-TTS-2-bf16.safetensors` | Same weights merged into one file (bit-exact). Fastest. |
| INT8 all linears | `Breeze-TTS-2-int8-convrot.safetensors` | Smallest build; slower in the decode loop (see below). |
| INT8 text encoder only | `Breeze-TTS-2-int8-text-encoder.safetensors` | Decode path fully bf16; smallest quality-irrelevant VRAM cut. |

All mirrors live in [drbaph/Breeze-TTS-2-comfyui](https://huggingface.co/drbaph/Breeze-TTS-2-comfyui).

## Model Files

If `C:/Models/audio/speech/TTS/Breeze-TTS-2/` contains the original sharded checkpoint, it is used directly for every model selection. Otherwise, weights are stored per source repo under `ComfyUI/models/breezetts2/` (primary folder first; extra_model_paths.yaml entries and symlinks are searched too). Missing files download automatically when `download_if_missing` is on, otherwise the error names the expected folder.

```
📂 ComfyUI/
└── 📂 models/
    └── 📂 breezetts2/
        └── 📂 drbaph_Breeze-TTS-2-comfyui/      (all builds, default source)
            ├── Breeze-TTS-2-bf16.safetensors              (6.49 GiB, official shards merged bit-exact)
            ├── Breeze-TTS-2-int8-hybrid.safetensors       (4.53 GiB, recommended int8 build)
            ├── Breeze-TTS-2-int8-convrot.safetensors      (4.22 GiB, all 462 linears quantized)
            ├── Breeze-TTS-2-int8-text-encoder.safetensors (5.84 GiB, decode path bf16)
            ├── config.json
            ├── generation_config.json
            ├── tokenizer.json
            ├── tokenizer_config.json
            ├── special_tokens_map.json
            └── 📂 audio_tokenizer/                        (Qwen3-TTS 12 Hz codec, shared)
                ├── model.safetensors                      (0.64 GiB)
                ├── config.json
                ├── configuration.json
                └── preprocessor_config.json
```

Only the selected weights file is downloaded; every build shares the same `config.json`, tokenizer files, and `audio_tokenizer/` codec.

| Component | bf16 build | int8 builds |
| --- | --- | --- |
| Model weights | 6.49 GiB | 4.22–5.84 GiB |
| `audio_tokenizer` codec | 0.64 GiB | unchanged |
| Tokenizer + configs | ~35 MB | unchanged |

Expect roughly **5.3–7.5 GiB VRAM** depending on build and dtype; AIMDO DynamicVRAM pages castable weights to keep live VRAM pressure low alongside other models.

## Benchmarks (RTX 5090, ComfyUI loader, sdpa)

| Build | avg RTF | Peak VRAM | Whisper WER vs prompt |
| --- | --- | --- | --- |
| bf16 | 5.68 | 7.51 GiB | baseline |
| int8-convrot (all linears) | 9.08 | 5.21 GiB | matches bf16 |
| **int8-hybrid (depth decoder bf16)** | **5.64** | **5.53 GiB** | matches bf16 |
| int8-text-encoder only | 5.34 | 6.82 GiB | matches bf16 |

Why full INT8 regresses: the depth decoder runs 15 skinny single-token GEMMs per 12.5 Hz frame
(≈1260 linear calls/frame). At M=1–2 the int8 per-call overhead (~0.1 ms quantize + dispatch +
allocations) dwarfs the GEMM savings, so those layers lose to plain bf16 `F.linear` (~0.03 ms).
The backbone contributes only ~196 calls/frame and the text encoder runs once at prefill, so
quantizing them is nearly free — that is exactly what the hybrid build does. Full measurement
data is in `benchmark_report.json` on the mirror repo.

## Requirements

- ComfyUI with **Transformers 4.57 or 5.3+**. On 5.x the text encoder uses the native
  `T5Gemma2TextEncoder` wired into flash attention through a custom-registered FA2 path
  (upstream blocks it over softcapping concerns this checkpoint does not have); on 4.x it
  uses the vendored upstream compat implementation, which also runs flash attention.
- torch/torchaudio from your ComfyUI install. `flash_attn` and `sageattention` are optional.
- `comfy-kitchen` ships with current ComfyUI and is only imported when an INT8 checkpoint is loaded.

## Model architecture (what actually runs)

- **Text encoder**: T5Gemma2 (26 layers, 1152 hidden) + linear projection into the backbone space.
- **Backbone**: Qwen3-style decoder (28 layers, 2048 hidden) predicting the first codebook token per
  12.5 Hz frame, with an audio-token embedding summed over the 16 codebooks.
- **Depth decoder**: 12-layer decoder generating codebooks 1–15 per frame with its own CFG.
- **Audio tokenizer**: vendored Qwen3-TTS 12 Hz codec (Mimi-style encoder + streaming decoder),
  24 kHz, 16 codebooks. Reference audio is encoded with the same codec.
- The in-checkpoint `codec_model` (a Mimi copy used only by the upstream training forward) is not
  built at inference.

The eager generation loop reimplements the official `FastBreezeStreamingRuntime` with all fast
stages disabled (DynamicCache instead of CUDA graphs) so it stays compatible with AIMDO dynamic
VRAM paging and INT8 weight casting. With `decode_mode=cuda_graphs` the depth decode steps are
captured into CUDA graphs instead — the captured decode runs **7–9x faster** than the eager
AIMDO-paged path (~3–4x faster end-to-end generation); that mode keeps the model weights
VRAM-resident rather than AIMDO-paged. The official CUDA-graph fast path is intentionally not ported.

## License

Node-pack code: Apache-2.0 (see `LICENSE`). Vendored model code keeps its upstream licenses
(Apache-2.0 from breezeblue-ai/breeze-tts and Qwen/Qwen3-TTS).

**Model weights, derivative checkpoints, and self-hosted outputs are governed by the BreezeBlue
Research and Non-Commercial License** — see [MODEL_LICENSE](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE).
Commercial use requires written authorization from RESONIA, INC.
