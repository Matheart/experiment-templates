
# TinyStories Transformer Training Template

This repository provides a highly optimized template for training small-scale Transformer models (e.g., 33M parameters) on the TinyStories dataset. It is designed to demonstrate maximum hardware utilization on modern GPUs (A100/H100).

## 🚀 Key Accelerations
To achieve ~15-minute epochs (down from ~2 hours), the following optimizations are critical:

### 1. Precision & Compute
- **Use `bfloat16` (BF16):** - **Benefit:** Halves VRAM usage and doubles compute throughput on Ampere+ GPUs compared to FP32.
  - **Code:** Wrap the forward pass in `with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):`.
- **Use TF32:** - Enable global TensorFloat-32 for internal matrix math: `torch.backends.cuda.matmul.allow_tf32 = True`.
  - *Speedup: ~2-3x over standard FP32.*

### 2. Memory & Attention
- **Flash Attention (SDPA):**
  - **Critical:** Standard attention creates an $O(N^2)$ memory bottleneck ($512 \times 512$ matrix), limiting batch size.
  - **Fix:** Use `attn_implementation="sdpa"` in the model config.
  - **Note:** Requires modern configs (e.g., `GPT2Config`, `LlamaConfig`). Avoid outdated architectures like `GPTNeoForCausalLM` which lack native SDPA support.
- **Vocabulary Size:**
  - Reducing vocab from 50k $\to$ 10k speeds up the final projection layer significantly (approx. 2x speedup for tiny models).

### 3. Compilation
- **`torch.compile`:**
  - **Issue:** For shallow models (e.g., 4 layers), Python interpreter overhead can consume 50% of runtime.
  - **Fix:** `model = torch.compile(model)` fuses layers into single kernels, removing overhead.

---

## ⚙️ Training Best Practices

### Hyperparameters
- **Learning Rate:** Use a cosine decay schedule.
- **Weight Decay:** Apply **only** to weights (matrix multiplications). 
  - **Exclude:** Bias, LayerNorm, and Embeddings.
- **Scaling Rule:** When scaling batch size, adjust learning rate using the square root rule:  
  $$\text{LR}_{new} \approx \text{LR}_{base} \times \sqrt{\frac{\text{BatchSize}_{new}}{\text{BatchSize}_{base}}}$$

### Optimization Strategy
- **Optimizer:** Use `optimizer.zero_grad(set_to_none=True)` to save memory bandwidth.
- **Batch Size:** - **Counter-Intuitive:** For 33M models on A100, standard batches (32, 128) are **too small** to saturate the GPU.
  - **Target:** With Flash Attention + BF16, push Batch Size to **1024+** to hide kernel launch overhead.

---

## 📊 Benchmarks (A100)

| Setup | Seq Length | Batch Size | Time per Epoch |
| :--- | :--- | :--- | :--- |
| **Naive (FP32)** | 512 | 128 | ~2 hours |
| **Optimized (BF16 + Compile)** | 512 | 256 | ~45 mins |
| **Fully Fused (BF16 + Compile + SDPA)** | 512 | 1024 | **~15 mins*** |

*\*Note: Assumes standard TinyStories (~460M tokens). For larger 1B+ token variations, expect ~35 mins.*

---

## ⚠️ Common Pitfalls

1.  **The "Architecture Trap":** Using `GPTNeo` or older Hugging Face models will error out with Flash Attention. Use `GPT2Config` instead.
2.  **The "Indent Bug":** Ensure `loss.backward()` is called **outside** the `autocast` context to avoid unnecessary type casting.
3.  **The "Small Batch" Trap:** If GPU utilization is low (<40%), your batch size is likely too small for the hardware, causing the GPU to wait for the CPU.

