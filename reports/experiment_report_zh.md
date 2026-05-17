# GPU-CC MoE 传输侧信道与开销实验报告

日期：2026-05-17  
实验机器：Alibaba Cloud H20 GPU 实例，GPU Confidential Computing 开启  
模型：`/root/models/Qwen3-30B-A3B-Instruct-2507`

## 1. 这项实验想回答什么问题

这项实验关注一个具体的安全问题：

> 在 GPU Confidential Computing 开启时，攻击者如果不能读取模型内部的 router logits、top-k expert id、hidden states，只能看到 CPU TEE 和 GPU TEE 之间传输路径上的外部特征，例如传输大小和外部时序，那么他能否推测出 MoE 模型每层、每个 token 实际选择了哪些 experts？

这里的结论不依赖“读取 top-k”本身。读取 top-k 只出现在一个隔离的 trusted label run 里，用来生成离线 ground truth 标签。真正模拟攻击者的 attacker feature run 不读取 router、logits、top-k 或 hidden states。

本轮实验的初步结论是：

> 当前单卡、BF16、Transformers/PyTorch、全模型驻留 GPU 的 Qwen3 MoE 推理路径下，没有观察到明确的 CPU-GPU 传输路径 top-k 泄露证据。主预测结果明显高于 label shuffle，但几乎等于长度匹配负控，因此更像是 prompt 长度、layer、token position 等先验造成的效果，而不是 transfer-only 特征本身泄露了 expert routing。

## 2. 必要背景

### 2.1 什么是 GPU Confidential Computing

Confidential Computing 的目标是保护“正在计算中的数据”。普通加密主要保护存储或网络传输中的数据，但模型推理时，输入、权重、中间结果会进入 CPU/GPU 内存和计算单元。GPU CC 试图让这些数据在受保护的环境中运行，减少云平台宿主机或其他非授权方直接观察数据的机会。

在本实验中，可以把系统粗略理解为两端：

- CPU TEE：CPU 侧的受保护虚拟机环境。
- GPU TEE：GPU 侧的受保护执行环境。

CPU 和 GPU 之间仍然需要传输数据。GPU CC 下，传输路径会受到保护，但普通 guest/CVM root 权限不能直接看到驱动内部真实的 bounce buffer 地址、GPU DMA 解密事件或宿主机 VFIO/QEMU 细节。因此本实验只做黑盒测量：能直接测的是端到端 copy 时间、CUDA event 时间、CPU 调用时间等；内部四阶段拆解只能作为估计。

### 2.2 什么是 MoE 和 top-k experts

MoE 是 Mixture of Experts。它不是让每个 token 都经过同一套完整 FFN，而是在每一层为每个 token 选择若干个 expert 子网络。Qwen3-30B-A3B-Instruct-2507 的配置为：

- `num_hidden_layers = 48`
- `hidden_size = 2048`
- `num_experts = 128`
- `num_experts_per_tok = 8`
- `moe_intermediate_size = 768`
- `torch_dtype = bfloat16`

也就是说，对每个 token，在每一层，router 会从 128 个 experts 中选出 top-8。这个 top-8 expert id 列表就是本实验要预测的标签。

### 2.3 什么是侧信道

侧信道不是直接读取秘密，而是从“副作用”推测秘密。例如：

- 某次请求的运行时间更长。
- CPU 到 GPU 的传输数据更大。
- 某些同步等待时间发生变化。

如果这些外部现象和内部 top-k expert 选择之间存在稳定相关性，攻击者就可能在不直接读取 router 输出的情况下推测 routing 结果。

## 3. 威胁模型和权限边界

本实验采用保守权限模型：

- 攻击者只有 guest/CVM root 级别观察能力。
- 攻击者不能读取 router logits。
- 攻击者不能读取 top-k expert id。
- 攻击者不能读取 hidden states。
- 攻击者不能观察宿主机 VFIO/QEMU 内部细节。
- 攻击者不能直接观察真实 bounce buffer 地址或 GPU DMA 解密事件。

因此，attacker feature run 只保留如下类型的特征：

- prompt 字符长度。
- 输入 token 数。
- 输出 token 数。
- tokenization CPU 时间。
- 输入 H2D 字节数。
- 输入 H2D 时间。
- generate CPU wall time。
- decode output CPU 时间。
- 输出 D2H 字节数。

这些特征是外部推理路径可观察信息，不包含模型内部 routing 信息。

## 4. 实验结构

代码被拆成三个独立 CLI，避免把性能插桩和安全结论混在一起。

### 4.1 `cc-path-bench`

用途：测 CPU TEE 到 GPU TEE 的传输路径开销。

输出：

- `cc_path_measurements.jsonl`
- `operator_roofline.csv`
- `env.json`

记录字段包括：

- H2D/D2H 方向。
- tensor size。
- CPU wall time。
- CUDA event time。
- 同步等待时间。
- driver call 近似时间。
- syscall CPU time。

并输出四段估计：

- `cpu_stage_encrypt_est_ms`
- `cpu_to_bounce_est_ms`
- `bounce_to_gpu_dma_est_ms`
- `gpu_decrypt_residual_est_ms`

这些四段字段全部标记为 `estimated`。这点很重要，因为当前权限下不能直接观测驱动内部加密、真实 bounce buffer 或 GPU DMA 解密事件。

### 4.2 `routing-sidechannel`

用途：安全实验。

它分成三个子命令：

1. `trusted-label-run`

   使用 router hook 读取每层每 token 的 top-8 experts，生成离线标签。

2. `attacker-feature-run`

   不读取 router/logits/top-k/hidden states，只记录外部传输和时序特征。

3. `evaluate`

   把 trusted labels 和 attacker features join，训练一个简单的多标签 KNN 预测器，评估是否能预测 top-k experts。

评估指标：

- top-k exact match。
- top-k Jaccard。
- micro-F1。
- per-expert AUC macro。
- per-expert F1 macro。

负控：

- label shuffle：打乱标签后再评估。如果模型仍然表现很好，说明评估设计有问题。
- length matched prompt：只用长度匹配先验做预测。如果主结果和这个负控接近，说明预测主要来自 prompt 长度、token position、layer 等先验，而不是 transfer 特征。

### 4.3 `moe-latency-instrument`

用途：性能可观测性。

它记录：

- tokenization。
- H2D。
- generate wall time。
- decode output。
- attention 模块时延。
- router 模块时延。
- MoE FFN 模块时延。

这些内部时延只用于性能分析，不用于证明 CPU-GPU 传输路径泄露。如果 timing-only 特征在未来显示出预测能力，应归类为 GPU 执行时序侧信道，而不是 CPU-GPU transfer path 泄露。

## 5. 实际运行环境

本轮实验环境：

- GPU：NVIDIA H20
- Driver：550.144.03
- CUDA runtime：12.4
- PyTorch：2.6.0+cu124
- Transformers：4.57.6
- Hugging Face Hub：0.36.2
- 模型权重目录：`/root/models/Qwen3-30B-A3B-Instruct-2507`
- 模型目录大小：约 57 GB
- GPU 显存：约 97 GB

注意：之前尝试过 `transformers 5.8.1`，它与环境中的 `huggingface_hub` API 不兼容。因此项目已把 GPU extra 约束为 `transformers>=4.57,<5`。

## 6. 本轮运行的数据集

为了先得到一版可解释结果，本轮使用 32 条长度接近的英文技术 prompt：

- prompt 文件：`prompts/sidechannel_32.txt`
- prompt 数：32
- token 长度范围：10 到 16
- 平均 token 长度：13.375

这些 prompt 被设计为长度接近，是为了降低“长 prompt 和短 prompt 天然不同”带来的混淆。

## 7. Side-Channel 实验结果

输出目录：

- labels：`runs/qwen32-labels/`
- features：`runs/qwen32-features/`
- evaluation：`runs/qwen32-eval/`

数据规模：

- `routing_labels.jsonl`：20,544 行
- `attacker_features.jsonl`：428 行
- join 后样本：20,544 行
- train：15,216 行
- test：5,328 行

20,544 行 labels 来自：

```text
32 prompts × 48 layers × 每个 prompt 的 input tokens
```

每行 label 是一个 top-8 expert id 列表。

### 7.1 主结果

```text
top-k exact match:      0.0593
top-k Jaccard:          0.3312
micro-F1:               0.4438
per-expert AUC macro:   0.8040
per-expert F1 macro:    0.4176
```

这些数值看起来高于随机，但不能直接解释为 transfer path 泄露，因为还必须和负控比较。

### 7.2 Label Shuffle 负控

```text
top-k exact match:      0.0000
top-k Jaccard:          0.0395
micro-F1:               0.0711
per-expert AUC macro:   0.5034
per-expert F1 macro:    0.0625
```

这个负控接近随机，说明 evaluator 本身没有在打乱标签后仍然产生虚假强信号。

### 7.3 Length Matched 负控

```text
top-k exact match:      0.0723
top-k Jaccard:          0.3198
micro-F1:               0.4294
per-expert AUC macro:   0.7915
per-expert F1 macro:    0.4043
```

主结果相对长度匹配负控的增量：

```text
top-k exact match:     -0.0130
top-k Jaccard:         +0.0114
micro-F1:              +0.0144
per-expert AUC macro:  +0.0125
per-expert F1 macro:   +0.0133
```

这个结果非常关键：主模型比 label shuffle 明显好，但几乎贴近 length matched control。也就是说，目前预测效果主要可以由 prompt 长度、token position、layer 等先验解释，不能归因于 CPU-GPU transfer-only 特征泄露了 top-k routing。

## 8. CC Path 和 Operator Benchmark 结果

输出目录：

- `runs/cc-path-h20-full/`

数据规模：

- `cc_path_measurements.jsonl`：200 行
- `operator_roofline.csv`：41 行

本轮测量的端到端 CPU wall time 有效吞吐如下：

```text
4KB   H2D: 0.033 GB/s
4KB   D2H: 0.030 GB/s
64KB  H2D: 0.487 GB/s
64KB  D2H: 0.470 GB/s
1MB   H2D: 3.719 GB/s
1MB   D2H: 3.702 GB/s
16MB  H2D: 6.984 GB/s
16MB  D2H: 9.521 GB/s
64MB  H2D: 8.999 GB/s
64MB  D2H: 10.432 GB/s
```

小 tensor 受固定开销影响很大，因此 4KB 和 64KB 的有效吞吐低。1MB 时约 3.7 GB/s，接近实验计划中用作 CC 路径参考的 4 GB/s 量级。更大 tensor 的有效吞吐更高，说明当前黑盒测量中还包含 PyTorch runtime、pinned memory、PCIe 传输、驱动路径和缓存/调度等多种因素，不能简单等同于内部加密链路带宽。

Operator roofline 覆盖：

- MoE 形状的 host-device tensor transfer。
- token ids。
- hidden states。
- router-like top-k index tensor。
- expert-dispatch-like gather/scatter buffer。
- Qwen3 MoE expert FFN 形状 BF16 GEMM。

GEMM sweep 使用：

```text
hidden_size = 2048
moe_intermediate_size = 768
num_experts = 128
num_experts_per_tok = 8
tokens_per_expert = 1, 4, 16, 64, 256
```

本轮测得 BF16 GEMM 最大 achieved throughput 约为 51.82 TFLOP/s。这个数值用于本机 roofline 校准，不应直接当作 H20 官方峰值。

## 9. 如何理解这次结果

### 9.1 为什么不能说 top-k 被传输路径泄露了

在这个默认单卡推理路径下：

- 模型权重常驻 GPU。
- expert routing 在 GPU 内部完成。
- top-k expert id 不需要作为 CPU-GPU 传输数据返回 CPU。
- attacker feature run 只看到外部输入/输出大小和时间。

因此，从系统结构上看，CPU-GPU transfer path 直接泄露 top-k 的先验概率本来就低。

实验结果也符合这个判断：主预测结果相对 label shuffle 很高，但相对 length matched control 只高一点点。这说明模型学到的主要不是“传输路径里的 secret”，而是“同类 prompt、相近长度、相同 layer/token position 下 routing 分布有相似性”。

### 9.2 如果 timing-only 以后变强，应该怎么归类

如果未来扩大实验后发现仅使用 generate wall time 或模块级 latency 就能预测 top-k，那也不能自动归类为 CPU-GPU transfer path 泄露。更合理的分类是：

> GPU execution timing side channel

只有当额外信息明确来自 H2D/D2H transfer size、direction、copy timing 等传输路径变量，并且超过长度匹配等负控，才应讨论 CPU-GPU transfer path leakage。

## 10. 当前局限


主要局限：

- prompt 数只有 32，需要扩大到更多 prompt 和更多主题。
- 当前 prompt 都是英文技术句子，分布较窄。
- 只跑了一个 train/test seed：17。
- attacker features 对同一个 prompt 的所有 token 基本相同，粒度较粗。
- 没有 CC-Off 对照，因此不能实测非 CC overhead。
- 没有 CC-DevTools，因此没有 Nsight/NVTX 或驱动内部 attribution。
- bounce buffer 和 GPU DMA 解密事件仍然是黑盒估计，不是直接观测。
- 当前 evaluator 是轻量 KNN，用于验证侧信道信号，不代表最强攻击模型。

## 11. 建议的下一步

为了把结论做得更稳，应继续做以下实验：

1. 扩大 prompt 数量到 256 或 1024，并保持长度分桶。
2. 对每个长度桶做独立 train/test split，避免长度先验主导结果。
3. 多 seed 重复评估，报告均值和置信区间。
4. 增加“仅 transfer size”、“仅 timing”、“transfer + timing”三组特征消融。
5. 增加 batch size sweep，观察 batching 是否引入新的可观察差异。
6. 增加 max_new_tokens sweep，区分 prefill 和 decode 影响。
7. 如果未来有 CC-Off 或 CC-DevTools 环境，再补充非 CC 对照和更细 attribution。

## 12. 复现实验命令

安装依赖：

```bash
conda run -n CC python -m pip install -e .
conda run -n CC python -m pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple "transformers>=4.57,<5" accelerate
```

生成 trusted labels：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run -n CC routing-sidechannel \
  --run-id qwen32-labels \
  trusted-label-run \
  --out runs/qwen32-labels \
  --model /root/models/Qwen3-30B-A3B-Instruct-2507 \
  --prompts prompts/sidechannel_32.txt \
  --trust-remote-code
```

生成 attacker features：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run -n CC routing-sidechannel \
  --run-id qwen32-features \
  attacker-feature-run \
  --out runs/qwen32-features \
  --model /root/models/Qwen3-30B-A3B-Instruct-2507 \
  --prompts prompts/sidechannel_32.txt \
  --max-new-tokens 1 \
  --trust-remote-code
```

评估侧信道：

```bash
conda run -n CC routing-sidechannel evaluate \
  --labels runs/qwen32-labels/routing_labels.jsonl \
  --features runs/qwen32-features/attacker_features.jsonl \
  --out runs/qwen32-eval \
  --neighbors 5 \
  --test-fraction 0.25 \
  --seed 17
```

运行 CC path benchmark：

```bash
conda run -n CC cc-path-bench \
  --out runs/cc-path-h20-full \
  --sizes 4KB,64KB,1MB,16MB,64MB \
  --repeat 20 \
  --warmup 3 \
  --operator-roofline \
  --io-token-counts 1,16,128,1024 \
  --tokens-per-expert 1,4,16,64,256 \
  --gemm-repeat 50
```

## 13. 参考资料

- NVIDIA Hopper Confidential Computing whitepaper: <https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/HCC-Whitepaper-v1.0.pdf>
- NVIDIA Confidential Computing Deployment Guide: <https://docs.nvidia.com/cc-deployment-guide-tdx-snp.pdf>
- NVIDIA H100 Confidential Computing technical blog: <https://developer.nvidia.com/blog/confidential-computing-on-h100-gpus-for-secure-and-trustworthy-ai/>
- Qwen3-30B-A3B-Instruct-2507 config: <https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/blob/01ee60ece4bb0c7a758003e9f45e4a9059b20594/config.json>
