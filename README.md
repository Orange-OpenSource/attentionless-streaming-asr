# attentionless_streaming_asr

## Overview
This directory contains a compact codebase for experimenting with "attentionless" streaming ASR architectures used in the paper **["Do we really need Self-Attention for Streaming Automatic Speech Recognition?"](https://arxiv.org/abs/2601.19960)**. It is focused on training, model definitions and dataset preparation utilities used in our experiments.

**What is in this folder**
- `train.py` — main training entrypoint.
- `librispeech_prepare.py`, `tedlium2_prepare.py` — dataset preparation helpers.
- `requirements.txt` — Python dependencies for running the code in this folder.
- `models/` — model implementations (`DeformableConformer.py`, `Transformer.py`, `TransformerASR.py`).
- `decoders/` — decoder implementations (for example `transducer.py`).
- `hparams/` — example configuration YAMLs (e.g. `conformer_transducer.yaml`, `deformable_conv.yaml`).
- `utils/` — utilities used by training (e.g. `train_logger.py`, loss helpers).

**Quick start**
1. install necessary apt packages:
```bash
sudo apt-get install ffmpeg python3.10 pip
```
2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```
3. Install dependencies from the local `requirements.txt`:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Basic usage**
- Inspect and adapt the example hyperparameters in `hparams/`.
- Run training (adjust arguments as needed):
    - Replacing Self-Attention with a deformable convolution (Soft Approach):
        ```bash
        python train.py hparams/deformable_conv.yaml
        ```
    - Removing Self-Attention without any replacement (Hard Approach)
        ```bash
        python train.py hparams/deformable_conv.yaml --mhdcn_kernel_size=-1
        ```
    - If you want to train the standard conformer:
        ```bash
        python train.py hparams/conformer_transducer.yaml
        ```
-Run inference (example with the Soft Approach):
    ```bash
    python train.py hparams/conformer_transducer.yaml --test_only
    ```

**Notes**
- `requirements.txt` is intended as the canonical dependency list for reproducibility — pin versions if you need stable runs.
- Models and decoders are implemented in `models/` and `decoders/`. Use `hparams/` to configure training settings.

**Contributing**
- Open an issue for questions or proposed changes.
- Submit PRs with clear descriptions and reproduction steps.

**License**
- MIT
