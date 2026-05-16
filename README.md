# GPU-CC MoE Inference Experiments

This repo implements three separate experiment surfaces for H20 confidential-computing MoE inference:

- `cc-path-bench`: CPU TEE to GPU TEE transfer path measurement and operator roofline benchmarks.
- `routing-sidechannel`: routing side-channel experiment using trusted top-k labels only as offline ground truth.
- `moe-latency-instrument`: performance instrumentation for model execution latency, not used as transfer-path leakage evidence.

The default model is guest/CVM root only. The code does not claim direct observation of real bounce-buffer addresses, GPU DMA decrypt events, or host VFIO/QEMU internals. The four-stage CC path fields are black-box estimates and are marked as estimated in every row.

## Environment Note

The target environment here is Alibaba Cloud H20 with CC-On and no CC-DevTools. Nsight/NVTX attribution is therefore optional and not required by the default scripts.

If `nvidia-smi` shows the H20 but `torch.cuda.is_available()` is false, check the PyTorch CUDA wheel against the installed driver. For example, a `+cu130` PyTorch wheel will not initialize against a CUDA 12.4 driver.

## CLI Layout

```bash
conda run -n CC python -m pip install -e .
conda run -n CC python -m pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple transformers accelerate

cc-path-bench --out runs/cc-on-transfer --repeat 20 --operator-roofline
routing-sidechannel trusted-label-run --out runs/labels --model Qwen/Qwen3-30B-A3B-Instruct-2507 --prompts prompts.txt
routing-sidechannel attacker-feature-run --out runs/features --model Qwen/Qwen3-30B-A3B-Instruct-2507 --prompts prompts.txt
routing-sidechannel evaluate --labels runs/labels/routing_labels.jsonl --features runs/features/attacker_features.jsonl --out runs/eval
moe-latency-instrument --out runs/latency --model Qwen/Qwen3-30B-A3B-Instruct-2507 --prompts prompts.txt
```

Expected output files:

- `env.json`
- `cc_path_measurements.jsonl`
- `routing_labels.jsonl`
- `attacker_features.jsonl`
- `sidechannel_eval.json`
- `operator_roofline.csv`
- `plots/*.png` when plotting dependencies are available

## Security Interpretation

`routing-sidechannel` separates the trusted label run from the attacker feature run. The attacker run does not read router logits, top-k tensors, or hidden states. In the default single-GPU BF16 Transformers/PyTorch path with the full model resident on GPU, expert routing is expected to remain inside GPU execution. Transfer-only features should therefore be close to random guessing. If timing-only features exceed random, classify that as a GPU execution timing side channel, not CPU-GPU transfer-path leakage.

## Local Tests

The local tests avoid CUDA-specific requirements:

```bash
conda run -n CC python -m unittest discover -s tests
```
