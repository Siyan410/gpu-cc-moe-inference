# GPU-CC MoE Inference Experiments

This repo contains black-box GPU confidential-computing experiments for H20
MoE inference. The code is organized as four separate command-line surfaces so
that transfer-path measurement, routing side-channel evaluation, performance
instrumentation, and memory-pressure behavior are not mixed together.

- `cc-path-bench`: CPU TEE to GPU TEE transfer-path measurements plus optional
  operator roofline benchmarks.
- `routing-sidechannel`: routing side-channel experiment. Trusted top-k labels
  are collected only as offline ground truth; the attacker feature run does not
  read router logits, top-k tensors, or hidden states.
- `moe-latency-instrument`: model execution latency instrumentation for
  performance analysis. These internal timings are not transfer-path leakage
  evidence.
- `memory-pressure-observe`: observes whether the current Transformers
  `device_map=auto` inference path remains GPU-resident or starts CPU/disk
  offload under artificial GPU memory pressure.
- `overhead-amplification`: runs matched prompt/request-shape workloads for
  CC-On versus CC-Off availability experiments and computes ratio-of-ratios.

The default threat model is guest/CVM root only. The code does not claim direct
observation of real bounce-buffer addresses, GPU DMA decrypt events, or host
VFIO/QEMU internals. The four-stage CC path fields are black-box estimates and
are marked as estimated in every row.

## Environment Note

The target environment here is Alibaba Cloud H20 with CC-On and no CC-DevTools.
Nsight/NVTX attribution is therefore optional and not required by the default
scripts.

If `nvidia-smi` shows the H20 but `torch.cuda.is_available()` is false, check
the PyTorch CUDA wheel against the installed driver. For example, a `+cu130`
PyTorch wheel will not initialize against a CUDA 12.4 driver.

## Install

```bash
conda run -n CC python -m pip install -e .
conda run -n CC python -m pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple "transformers>=4.57,<5" accelerate
```

## CLI Layout

```bash
cc-path-bench \
  --out runs/cc-path-h20-full \
  --repeat 20 \
  --operator-roofline

routing-sidechannel \
  --run-id qwen32-labels \
  trusted-label-run \
  --out runs/qwen32-labels \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --prompts prompts/sidechannel_32.txt

routing-sidechannel \
  --run-id qwen32-features \
  attacker-feature-run \
  --out runs/qwen32-features \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --prompts prompts/sidechannel_32.txt

routing-sidechannel evaluate \
  --labels runs/qwen32-labels/routing_labels.jsonl \
  --features runs/qwen32-features/attacker_features.jsonl \
  --out runs/qwen32-eval

moe-latency-instrument \
  --run-id qwen-smoke-latency \
  --out runs/qwen-smoke-latency \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --prompts prompts/smoke.txt \
  --max-new-tokens 1

memory-pressure-observe \
  --run-id qwen-memory-pressure-load-only \
  --out runs/qwen-memory-pressure-load-only \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --prompts prompts/smoke.txt \
  --reserve-gb 0,35,45

overhead-amplification run-transformers \
  --run-id overhead-on-pilot \
  --out runs/overhead-on-pilot \
  --model /root/models/Qwen3-30B-A3B-Instruct-2507 \
  --cc-mode on \
  --suite pilot \
  --batch-sizes 1 \
  --repeats 3 \
  --trust-remote-code

overhead-amplification compare \
  --cc-on runs/overhead-on-pilot \
  --cc-off runs/overhead-off-pilot \
  --out runs/overhead-compare-pilot
```

## Output Files

Common output:

- `env.json`

`cc-path-bench` output:

- `cc_path_measurements.jsonl`
- `operator_roofline.csv` when `--operator-roofline` is enabled

`routing-sidechannel` output:

- `routing_labels.jsonl` from `trusted-label-run`
- `attacker_features.jsonl` from `attacker-feature-run`
- `sidechannel_eval.json` from `evaluate`

`moe-latency-instrument` output:

- `moe_latency_measurements.jsonl`

`memory-pressure-observe` output:

- `memory_pressure_observations.jsonl`

`overhead-amplification run-transformers` output:

- `overhead_measurements.jsonl`
- `overhead_summary.json`

`overhead-amplification compare` output:

- `overhead_comparison.json`
- `overhead_comparison.csv`

Plot files under `plots/*.png` may be added when plotting dependencies and
plotting scripts are present. They are not required by the core CLIs.

## Existing Results

The current workspace already contains run artifacts under `runs/` and detailed
writeups under `reports/`.

### Routing Side Channel

Main run directories:

- `runs/qwen32-labels/`
- `runs/qwen32-features/`
- `runs/qwen32-eval/`

Observed data volume:

- `routing_labels.jsonl`: 20,544 rows
- `attacker_features.jsonl`: 428 rows
- joined evaluation rows: 20,544
- train rows: 15,216
- test rows: 5,328

Main evaluation in `runs/qwen32-eval/sidechannel_eval.json`:

```text
top-k exact match:      0.0593
top-k Jaccard:          0.3312
micro-F1:               0.4438
per-expert AUC macro:   0.8040
per-expert F1 macro:    0.4176
```

The label-shuffle negative control is close to random, while the length-matched
prompt control is close to the main result. The current interpretation is that
the observed prediction quality is mostly explained by prompt length, token
position, layer, and routing prior effects, not by CPU-GPU transfer-only leakage
of top-k routing.

### CC Path and Operator Benchmark

Main run directory:

- `runs/cc-path-h20-full/`

Observed data volume:

- `cc_path_measurements.jsonl`: 200 rows
- `operator_roofline.csv`: 41 rows

Average CPU wall-time effective throughput from the current run:

```text
4KB   H2D: 0.033 GB/s
4KB   D2H: 0.030 GB/s
64KB  H2D: 0.490 GB/s
64KB  D2H: 0.473 GB/s
1MB   H2D: 3.737 GB/s
1MB   D2H: 3.742 GB/s
16MB  H2D: 6.984 GB/s
16MB  D2H: 9.523 GB/s
64MB  H2D: 9.000 GB/s
64MB  D2H: 10.433 GB/s
```

The measured BF16 GEMM roofline sweep currently reaches about 51.82 TFLOP/s on
this setup. This is a local calibration value, not an official H20 peak claim.

### Latency Instrumentation

Current smoke run directory:

- `runs/qwen-smoke-latency/`

Observed data volume:

- `moe_latency_measurements.jsonl`: 145 rows

This smoke run contains one end-to-end row plus attention, router, and MoE FFN
module timing rows. It is performance instrumentation only.

### Memory Pressure

Main run directories:

- `runs/qwen-memory-pressure-load-only/`
- `runs/qwen-memory-pressure-baseline-generate/`

Observed behavior:

- 0 GiB reservation: model loads GPU-resident.
- 35 GiB reservation: `device_map=auto` loads with CPU offload.
- 45 GiB reservation: more modules are offloaded to CPU.
- 0 GiB baseline generate: one-token generate completed in about 1.52 s in the
  recorded run.

The directory `runs/qwen-memory-pressure/` currently contains only `env.json`,
so it appears to be an incomplete or setup-only run.

### Prompt-Induced Overhead Amplification

The `overhead-amplification` CLI is intended for paired CC-On and CC-Off runs.
Run the same command on both machines and change only `--cc-mode` plus the
output directory. The built-in suites are:

- `smoke`: short functional check.
- `pilot`: first useful run for baseline, decode-heavy, JSON-prompt,
  long-context, and streaming-step workloads.
- `full`: larger output and context sweep for stronger p95/p99 evidence.

Convenience script for the current Alibaba Cloud model path:

```bash
CC_MODE=on bash scripts/run_overhead_pilot.sh
CC_MODE=off bash scripts/run_overhead_pilot.sh
```

Optional environment variables:

```bash
SUITE=smoke|pilot|full
BATCH_SIZES=1,8
REPEATS=5
MEASURE_TTFT_PROXY=1
MODEL=/root/models/Qwen3-30B-A3B-Instruct-2507
OUT=runs/custom-output-dir
```

After copying both run directories onto one machine:

```bash
overhead-amplification compare \
  --cc-on runs/overhead-on-pilot \
  --cc-off runs/overhead-off-pilot \
  --out runs/overhead-compare-pilot
```

The main quantity to inspect is `ratio_of_ratios.latency_ms` or
`ratio_of_ratios.tpot_ms`:

```text
(CC-On stress / CC-Off stress) / (CC-On baseline / CC-Off baseline)
```

A value above 1 means the workload amplifies CC-specific overhead relative to
the normal baseline workload.

## Security Interpretation

`routing-sidechannel` separates trusted label collection from attacker feature
collection. The trusted run reads router top-k only to generate offline ground
truth. The attacker run does not read router logits, top-k tensors, or hidden
states.

In the default single-GPU BF16 Transformers/PyTorch path with the full model
resident on GPU, expert routing is expected to remain inside GPU execution.
Transfer-only features should therefore be close to random guessing after
controlling for prompt length and routing priors. If timing-only features exceed
random in future experiments, classify that as a GPU execution timing side
channel, not CPU-GPU transfer-path leakage.

Memory-pressure runs are a separate condition. Once `device_map=auto` triggers
CPU offload, the inference path is no longer the same GPU-resident path and may
introduce additional CPU-GPU transfers, waits, and synchronization effects.

## Reports

- `reports/experiment_report_zh.md`: full Chinese experiment report covering
  the side-channel, CC path, roofline, latency, and memory-pressure results.
- `reports/memory_pressure_observation_zh.md`: focused Chinese report for the
  memory-pressure observations.

## Local Tests

The local tests avoid CUDA-specific requirements:

```bash
conda run -n CC python -m unittest discover -s tests
```
