# Installation

Run commands from the repository root. Use an isolated environment for fusion and a separate one for downstream evaluation.

## CPU (Windows or Linux)

Python 3.12 was used for the release checks. Python 3.10 is also supported by the source.

```bash
python -m venv .venv
# Linux:
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The native PyTorch selective scan supports inference and gradients on CPU. Full-resolution inference is considerably slower than the CUDA kernel; CPU is useful for checking installation and small examples.

## CUDA (Linux, Python 3.10)

Use Python 3.10 and a driver compatible with CUDA 12.1. The following wheels match PyTorch 2.4 and the Linux CPython 3.10 ABI:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install 'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.3.post1/causal_conv1d-1.5.3.post1%2Bcu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
python -m pip install 'https://github.com/state-spaces/mamba/releases/download/v2.3.0/mamba_ssm-2.3.0%2Bcu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
```

Set `--device cuda:0 --scan-backend cuda` to require the original selective-scan kernel. `--scan-backend auto` selects the CUDA kernel for CUDA inputs and the PyTorch recurrence for CPU inputs. CUDA inputs require a working `mamba-ssm` installation unless `reference` is selected explicitly. Changing precision or the scan implementation can affect floating-point results. Release validation was performed on CPU; the CUDA comparison test runs when CUDA and `mamba-ssm` are available.

## Download weights

```bash
python -m tools.download_weights
python tools/verify_weights.py --root .
```

This downloads the final fusion state, SegFormer-B2 configuration and weights, the exact YOLOv8s prior, and the fixed FMB/M3FD downstream evaluators. Every file is SHA256-checked. Downloads can be restricted with `--only fusion`, `--only priors`, or `--only evaluators`. Existing files are verified and reused. `tools/verify_weights.py` verifies the fusion checkpoint; the downloader also verifies priors and evaluators when reusing their files.

The downloader obtains third-party priors directly from their authors. A newer YOLO checkpoint with the same filename can have different bytes; use the pinned version in the downloader. See [dependencies](DEPENDENCIES.md).

## Check installation

```bash
python -m safir.infer --help
python -m safir.train --help
python -m pip install pytest==8.3.3
python -m pytest -q
```

For the full weight-loading test, set `SAFIR_TEST_CHECKPOINT`, `SAFIR_TEST_SEGFORMER`, and `SAFIR_TEST_YOLO` to the three downloaded paths before running pytest. Without these variables the integration test is skipped; numerical tests still run.
