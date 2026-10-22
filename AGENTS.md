# Breeze fork development

Start from committed `main` in a Git CLI task worktree. Run
`COMFYUI_PYTHON=<prepared-python> CUTLERY_NODES_ROOT=<cutlery-checkout> python3 tools/setup_worktree.py`
before implementation. Python >=3.10, PyTorch, Torchaudio, NumPy and Faster-Whisper
must already be installed in the ComfyUI runtime; provision them separately using
the deployment's requirements. Setup checks imports and the explicit Cutlery source
path without installing packages, changing pins, loading models or contacting services.
Compatible runtime versions are accepted; rerunning setup preserves tracked files.

Run `<prepared-python> -m unittest discover -s tests -v` for isolated bridge and
speaker tests. Live tests require separately provisioned CTranslate2 weights under
ComfyUI's configured model root. Never download models during node execution.
Only restart ComfyUI after checking native, router and Workflows5 activity.

Use managed GitNexus `index_ensure` for the exact checkout with embeddings disabled
after setup and refresh it before graph-dependent review. Do not register task
worktrees as shared skill sources.

The shared `cutlery-comfyui-nodes` skill owns managed Whisper and this bridge's
operating guidance. Locate its authoritative repository-owned source with
`skills show cutlery-comfyui-nodes`; update that source for reusable changes,
then run `skills reconcile` and `skills doctor` after integration. The upstream
root `SKILL.md` is a dialogue-writing guide, not the Whisper lifecycle owner.
