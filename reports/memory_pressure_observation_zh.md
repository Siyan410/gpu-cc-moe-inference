# 显存不足时当前推理方案的行为观测

日期：2026-05-17  
模型：`/root/models/Qwen3-30B-A3B-Instruct-2507`  
GPU：NVIDIA H20，约 97GB 显存  
当前推理方案：Transformers `AutoModelForCausalLM.from_pretrained(..., device_map="auto", torch_dtype=bf16)`

## 1. 观测目的

本观测想回答：

> 当 H20 显存不足以完整容纳 Qwen3 MoE 模型时，当前推理方案会怎么做？是直接 OOM，还是自动把部分层放到 CPU，或者出现其他行为？

为了模拟显存不足，本实验在同一个 Python 进程中先分配 GPU 占位 tensor，人为减少可用显存，然后再用当前推理方案加载模型。

## 2. 新增工具

新增 CLI：

```bash
memory-pressure-observe
```

输出文件：

```text
runs/<id>/env.json
runs/<id>/memory_pressure_observations.jsonl
```

默认行为：

- 先按 `--reserve-gb` 指定大小占用 GPU 显存。
- 再加载模型。
- 记录 `hf_device_map`、显存快照、是否发生 CPU/disk offload。
- 默认不执行 generate，因为一旦模型被 offload 到 CPU，单 token 生成可能非常慢。
- 如需测正常生成耗时，可加 `--run-generate`。

## 3. 运行命令

显存压力 load-only sweep：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run -n CC memory-pressure-observe \
  --run-id qwen-memory-pressure-load-only \
  --out runs/qwen-memory-pressure-load-only \
  --model /root/models/Qwen3-30B-A3B-Instruct-2507 \
  --prompts prompts/smoke.txt \
  --reserve-gb 0,35,45 \
  --trust-remote-code
```

正常 0GiB baseline 生成：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run -n CC memory-pressure-observe \
  --run-id qwen-memory-pressure-baseline-generate \
  --out runs/qwen-memory-pressure-baseline-generate \
  --model /root/models/Qwen3-30B-A3B-Instruct-2507 \
  --prompts prompts/smoke.txt \
  --reserve-gb 0 \
  --run-generate \
  --max-new-tokens 1 \
  --trust-remote-code
```

## 4. 观测结果

### 4.1 0GiB reservation：全模型上 GPU

```text
status: loaded_generate_not_requested
offload: False
device_counts: {'0': 1}
model_load_ms: 22454.8
```

显存快照：

```text
before_reservation: used 0.57 GiB, free 93.97 GiB
after_model_load:   used 57.45 GiB, free 37.08 GiB
after_cleanup:      used 0.57 GiB, free 93.97 GiB
```

解释：没有人为显存压力时，`device_map=auto` 将整个模型作为一个 root module 放到 GPU 0。模型加载后占用约 57.45GiB。

### 4.2 35GiB reservation：自动 CPU offload

```text
status: loaded_with_offload_generate_skipped
offload: True
device_counts: {'0': 45, 'cpu': 7}
model_load_ms: 23000.2
```

显存快照：

```text
before_reservation: used 0.57 GiB,  free 93.97 GiB
after_reservation:  used 35.57 GiB, free 58.97 GiB
after_model_load:   used 87.23 GiB, free 7.30 GiB
after_cleanup:      used 0.57 GiB,  free 93.97 GiB
```

部分 `hf_device_map`：

```text
lm_head: cpu
model.embed_tokens: 0
model.layers.0: 0
...
model.layers.47: cpu
model.norm: cpu
model.rotary_emb: cpu
```

Transformers 同时输出提示：

```text
Some parameters are on the meta device because they were offloaded to the cpu.
```

解释：当先占用 35GiB 显存后，GPU 剩余约 59GiB。当前 `device_map=auto` 没有直接 OOM，而是把部分模块放到 CPU。GPU 仍被尽量填满，加载后只剩约 7.3GiB。

### 4.3 45GiB reservation：更多 CPU offload

```text
status: loaded_with_offload_generate_skipped
offload: True
device_counts: {'0': 37, 'cpu': 15}
model_load_ms: 20122.0
```

显存快照：

```text
before_reservation: used 0.57 GiB,  free 93.97 GiB
after_reservation:  used 45.57 GiB, free 48.97 GiB
after_model_load:   used 87.95 GiB, free 6.58 GiB
after_cleanup:      used 0.57 GiB,  free 93.97 GiB
```

解释：当先占用 45GiB 显存后，GPU 剩余约 49GiB，低于完整 GPU 驻留所需空间。当前 `device_map=auto` 继续选择 CPU offload，而不是直接 OOM。相比 35GiB，更多模块被放到 CPU。

## 5. 正常 baseline 生成耗时

在无显存压力、全模型 GPU 驻留时，执行 1 token generate：

```text
status: ok
offload: False
model_load_ms: 22870.4
input_h2d_ms: 0.2812
generate_ms: 1524.1
output_tokens: 11
```

显存：

```text
after_model_load: used 57.45 GiB
after_generate:   used 57.61 GiB
```

解释：全 GPU 驻留时，短 prompt 生成 1 个新 token 的耗时约 1.5 秒。这个数值包括 Transformers generate 框架开销和当前 CC 环境下的执行开销。

## 6. 关键结论

当前推理方案在显存不足时的行为是：

1. `device_map=auto` 会尽量把模型放到 GPU。
2. 当 GPU 剩余显存不足时，它不会立刻 OOM，而是自动把一部分模块放到 CPU。
3. 35GiB reservation 时，device map 中已有 7 个模块在 CPU。
4. 45GiB reservation 时，device map 中有 15 个模块在 CPU。
5. GPU 显存会被尽量填到接近上限，保留约 6 到 7GiB 空闲。
6. 一旦出现 CPU offload，生成可能变得非常慢，且不再代表“全模型驻留 GPU”的默认高性能路径。

## 7. 对原安全实验的影响

这点对 side-channel 实验很重要：

- 原始安全假设是“全模型驻留 GPU，expert routing 在 GPU 内完成”。
- 当显存不足触发 CPU offload 后，这个假设不再完全成立。
- CPU offload 可能引入新的 CPU-GPU 传输、CPU 执行等待和跨设备同步。
- 因此，显存不足场景下观测到的 timing 或 transfer 差异，不能直接和全 GPU 驻留场景混在一起解释。

建议后续把两类实验分开命名：

- `gpu-resident`: 全模型 GPU 驻留。
- `memory-pressure-offload`: 显存压力触发 CPU offload。

如果要研究 offload 场景的安全性，应单独记录 `hf_device_map`，并把 CPU offload 本身作为实验条件，而不是把它当作同一个推理方案下的噪声。

## 8. 输出文件

本次观测输出：

```text
runs/qwen-memory-pressure-load-only/env.json
runs/qwen-memory-pressure-load-only/memory_pressure_observations.jsonl
runs/qwen-memory-pressure-baseline-generate/env.json
runs/qwen-memory-pressure-baseline-generate/memory_pressure_observations.jsonl
```
