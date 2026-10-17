# 🧠 Running a 35B MoE Model on Consumer Hardware

**Qwen3.6-35B-A3B · BGE-M3 · BGE-Reranker-V2-M3** — running locally with llama.cpp, zero cloud, zero API. Embedding and reranking models configured for RAG in Open WebUI.

[![MoE](https://img.shields.io/badge/MoE-35B_params-blue)](https://github.com)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-latest-green)](https://github.com/ggerganov/llama.cpp)
[![Hardware](https://img.shields.io/badge/Hardware-Consumer-orange)](https://github.com)

---

## 🚀 At a Glance

| | |
|---|---|
| **Main Model** | Qwen3.6-35B-A3B (MoE, 35B total / ~3B active per token) |
| **Hardware** | Ryzen 9 · RTX 2070 8 GB VRAM · 32 GB DDR5 |
| **Measured Perf** | ~27 tok/s generation |
| **Context Window** | 131,072 tokens |
| **Vision** | Yes (via mmproj on CPU) |
| **Embeddings** | BGE-M3 — multilingual, for RAG in Open WebUI |
| **Reranking** | BGE-Reranker-V2-M3 — refines retrieval results |

> 💡 **The secret**: `-np 1` — llama.cpp defaults `n_parallel` to 4, which quietly 4× your KV cache memory and OOMs you. Force a single slot and the model fits in 8 GB VRAM.

---

## 🖥️ Hardware Requirements

### Minimum
| Component | Specification |
|---|---|
| GPU | NVIDIA with ≥ 6 GB VRAM (RTX 2070 minimum) |
| System RAM | ≥ 16 GB DDR4/DDR5 |
| CPU | ≥ 8 cores (for MoE experts on CPU) |

### Recommended (this config)
| Component | Specification |
|---|---|
| GPU | NVIDIA with ≥ 8 GB VRAM (RTX 2070 / 3060 / 4060) |
| System RAM | ≥ 32 GB DDR5 |
| CPU | ≥ 12 cores (Ryzen 9 / Core i7+) |

---

## 📐 Model Architecture

### Qwen3.6-35B-A3B — The Heavy MoE

Qwen3.6-35B-A3B is a **Mixture of Experts** model from the Qwen family:
- **~35 billion** total parameters (the "A3B" suffix = ~3 billion active per token)
- Only **~3 billion** are computed at each forward pass thanks to MoE routing — the rest stay dormant until needed
- Native vision via separate multimodal projector (`mmproj`)
- Context support up to 150K tokens

The key insight: a "35B" model doesn't need 35B params doing work every step. The MoE architecture gates which experts are activated, so the effective compute is closer to a 3–4B dense model — while still retaining the knowledge of all 35B parameters for retrieval.

### BGE-M3 — Embeddings for RAG (Open WebUI)

Embedding model from BAAI, used by Open WebUI to convert your documents and queries into vector representations:
- Native support for Chinese, English, and other languages
- Context capacity: 8192 tokens per document chunk
- Used at ingest time to embed your knowledge base files, and at query time to find the most similar chunks

### BGE-Reranker-V2-M3 — Reranking for RAG (Open WebUI)

Reranking model from BAAI, used by Open WebUI to refine retrieval results after initial embedding search:
- Takes a query + document pair and scores their relevance precisely
- Pooling mode `rank` for pairwise comparison scoring
- Dramatically improves answer quality in RAG — the embedding step gives you candidates, the reranker tells you which ones actually matter

### How It All Fits Together in Open WebUI

```
User asks a question
  → BGE-M3 embeds the query into a vector
  → Vector search finds top-N matching document chunks (embedding similarity)
  → BGE-Reranker re-scores those N chunks for relevance ranking
  → Top reranked chunks are injected as context into Qwen3.6-35B-A3B's prompt
  → Model answers using the retrieved knowledge + its own training
```

---

## ⚙️ Detailed Configuration

### `presets.ini` — The Core Config

```ini
[qwen3.6-35b-a3b]
model = [path-to-Qwen3.6-38ba3b.gguf]
mmproj = [path-to-mmproj-F16.gguf]
jinja = on
ngl = 99
n-cpu-moe = 36
ctx-size = 131072
cache-type-k = q4_0
cache-type-v = q4_0
parallel = 1
flash-attn = on
sleep-idle-seconds = 90

# ... generation parameters (temperature, top-p, etc.)

[bge-m3]
model = [path-to bge-m3-Q8_0.gguf]
embedding = true
ctx-size = 8192
sleep-idle-seconds = 90

[bge-reranker-v2-m3]
model = [path-to bge-reranker-v2-m3-Q8_0.gguf]
rerank = true
embedding = true
pooling = rank
sleep-idle-seconds = 90
```

#### Explanation of Every Flag

##### Model Loading & Architecture

| Flag | Value | What it does | Why it matters |
|---|---|---|---|
| `model` | path to `.gguf` file | Specifies the model weights file | Point this to your downloaded Qwen3.6-35B-A3B GGUF file (Q4_K_XL or similar) |
| `mmproj` | path to multimodal projector | Loads the vision encoder for image understanding | Required if you want to send images with your prompts. Runs on CPU by default → ~0 extra VRAM cost |
| `jinja` | `on` / `off` | Enables Jinja2 chat template rendering | Qwen3.6 uses a custom chat format; this ensures proper system/user/assistant message templating |

##### GPU & Memory Offloading

| Flag | Value | What it does | Why it matters |
|---|---|---|---|
| `ngl` (n_gpu_layers) | 99 | Number of transformer layers to offload to GPU. `99 = "offload everything that fits"` | The more layers on GPU, the faster inference. Set as high as possible without OOMing — 99 is a safe max for 8 GB VRAM with this model |
| `n-cpu-moe` | 36 | **Number of MoE expert layers pushed to system RAM** instead of GPU. This is NOT core count — it's the number of expert FFN blocks offloaded to CPU memory | **The most important flag**. Each transformer layer in a MoE model contains multiple "experts" (smaller feed-forward networks). By default they live on GPU, eating VRAM. Pushing them to system RAM frees up precious GPU space for layers and KV cache. Find the right value by trial: start low and increase until OOM errors disappear |

##### Context & KV Cache

| Flag | Value | What it does | Why it matters |
|---|---|---|---|
| `ctx-size` | 131072 | Maximum context window in tokens (128K) | Huge context means you can feed long documents, codebases, or conversations. But KV cache scales linearly with ctx-size — hence the need for quantization below |
| `cache-type-k` | `q4_0` | Quantization format for the K (key) KV cache tensors | Q4_0 = 4-bit quantized. Cuts KV cache memory by ~2× compared to F16. Critical for fitting large context in limited VRAM |
| `cache-type-v` | `q4_0` | Quantization format for the V (value) KV cache tensors | Same as above — both K and V must be quantized together. The accuracy loss from Q4 vs F16 is negligible at 27 tok/s on consumer hardware |

##### Parallelism & Attention

| Flag | Value | What it does | Why it matters |
|---|---|---|---|
| `parallel` | 1 | Number of concurrent request slots. llama.cpp defaults to **4** | **The secret flag**. Default ×4 KV cache = OOM on 8 GB. Force to 1 and the full 128K context fits in 8 GB VRAM |
| `flash-attn` | `on` / `off` | Enables FlashAttention-2 for faster, more memory-efficient attention computation | Reduces both compute time and memory footprint during attention operations. Significant speedup on Ampere+ GPUs; still beneficial on Turing (RTX 20xx) |

##### Generation Parameters

| Flag | Value | What it does | Why it matters |
|---|---|---|---|
| `temperature` | 0.6 | Controls randomness in token sampling. Lower = more deterministic, higher = more creative | 0.6 is a sweet spot for code and technical tasks — coherent but not repetitive. For creative writing try 0.8–1.0; for factual/QA try 0.3–0.5 |
| `top-p` (nucleus sampling) | 0.95 | Only consider tokens whose cumulative probability reaches p | 0.95 means the top tokens that cover 95% of probability mass are candidates. Higher = more diverse, lower = safer/more focused |
| `top-k` | 20 | Restrict token sampling to the k most likely candidates at each step | Complements top-p: even if a tail token has decent cumulative prob, it won't be sampled unless it's in the top-20. Tightens output quality |
| `min-p` | 0.05 | Absolute floor — rejects any token below (min-p × top_token_prob) | A cleaner alternative to top-k filtering. 0.05 means no token gets picked if it has less than 5% the probability of the most likely token. Helps with incoherent outputs |
| `repeat-penalty` | 1.1 | Penalizes tokens that have appeared recently in the output | 1.1 = mild penalty. Prevents simple loops ("the the the") without killing fluency. Typical range: 1.0 (off) to 1.5 (strong) |
| `presence-penalty` | 0.0 | Penalizes tokens based on whether they've appeared *anywhere* in the output so far | 0 = disabled. Useful when you want to encourage the model to mention new topics/entities rather than rehashing old ones |
| `frequency-penalty` | 0.1 | Similar to presence penalty but weighted by how often a token has been used | Slight positive value encourages vocabulary diversity. 0.1 is gentle; values above 0.5 can make output feel unnatural |

##### Batch & Throughput Parameters

| Flag | Value | What it does | Why it matters |
|---|---|---|---|
| `ubatch-size` (micro-batch) | 2048 | Size of micro-batches for prompt processing. llama.cpp splits large batches into these chunks internally during prefill | Controls the granularity of KV cache computation when loading a long prompt. Smaller = lower peak memory but slightly more overhead. 2048 is a good default — balances throughput and VRAM usage |
| `batch-size` (logical batch) | 4096 | Number of tokens processed per decoding step across all active slots | Determines how many tokens the model processes in parallel during generation. Higher = better throughput but more compute pressure. On an RTX 2070, 4096 is a reasonable ceiling before hitting memory bandwidth limits |

##### Service Parameters

| Flag | Value | What it does | Why it matters |
|---|---|---|---|
| `sleep-idle-seconds` | 90 | Seconds of inactivity before the model unloads from GPU RAM | Prevents VRAM leaks and frees memory when no requests come in for 90 seconds. Essential for a dedicated server that may sit idle between uses |

##### BGE Models Specific Flags (RAG)

| Flag | Value | What it does |
|---|---|---|
| `embedding` | `true` | Enables embedding mode — the model outputs vector representations instead of text generations. Used by Open WebUI for document ingestion and query matching |
| `rerank` | `true` (BGE-Reranker only) | Enables reranking mode — takes query+document pairs and scores relevance. Used by Open WebUI to refine retrieval results after the initial embedding search |
| `pooling` | `rank` | Pooling strategy for the reranker. `rank` = pairwise ranking (query vs document). Other options include `mean`, `last`, etc. |

---

## 🏗️ Deployment with systemd

The file `llama-server.service` runs the server as a background daemon:

```ini
[Unit]
Description=llama.cpp server
After=network.target

[Service]
Type=simple
User=[username]
WorkingDirectory=[llamacpp installation directory]
ExecStart=[llama-server binary path] \
  --models-preset [path to presets.ini] \
  --models-max 1 \
  --models-autoload \
  --host 0.0.0.0 --port [port]
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### Setup Steps

```bash
# Copy the service file
sudo cp llama-server.service /etc/systemd/system/llama-server.service

# Edit and replace placeholders:
#   [username]          → your Linux username
#   [llamacpp installation directory] → where llama.cpp is built
#   [llama-server binary path]        → full path to llama-server binary
#   [path to presets.ini]             → absolute path to this config file
#   [port]                            → desired port (e.g., 8080)

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable --now llama-server.service

# Check status
systemctl status llama-server.service
```

### Test with curl

```bash
curl http://localhost:[port]/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-35b-a3b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.6,
    "max_tokens": 1024
  }'
```

---

## 📊 Benchmarks & Performance

### On this hardware (Ryzen 9 · RTX 2070 8 GB · 32 GB DDR5)

| Metric | Observed Value |
|---|---|
| Tokens/sec (generation) | ~27 tok/s |
| VRAM used | ~6.5–7.5 GB on RTX 2070 |
| System RAM | ~8–12 GB (depends on context size and active MoE experts) |
| First-token latency | ~1–3s (prompt loading + prefill) |

### Comparison with other known configs

| Hardware | VRAM | tok/s | Source |
|---|---|---|---|
| RTX 2070 8 GB (this config) | 8 GB | ~27 tok/s | Author |
| RTX 3060 12 GB | 12 GB | ~45 tok/s | [DogukanUrker](https://x.com/dogukanurker/status/2077472690156511643) |
| RTX 4090 24 GB | 24 GB | ~80+ tok/s | llama.cpp estimation |

> ⚠️ Performance varies with prompt length, temperature, and the number of MoE experts activated per token.

---

## ❓ FAQ / Troubleshooting

### Model won't start (OOM)
- Make sure `parallel = 1` is set in `[qwen3.6-35b-a3b]`
- Confirm `cache-type-k` and `cache-type-v` are both `q4_0`
- Increase `n-cpu-moe` gradually — it controls how many MoE expert layers go to RAM, not core count. Start low and increase until OOM errors disappear

### Responses are slow (> 10 tok/s latency)
- Verify GPU layers loaded correctly (`ngl = 99`)
- Reduce `ctx-size` if you don't need 128K tokens (KV cache scales linearly)
- First token is always slower due to prompt prefill — subsequent tokens are fast

### Vision doesn't work
- Check that the `mmproj` file exists at the specified path
- Don't set `--no-mmproj-offload` if you have little VRAM — leave it on CPU by default (it costs ~0 extra VRAM)

### RAG / BGE models not working in Open WebUI
- Make sure both `.gguf` files are downloaded and paths point to the correct locations
- The embedding model (`bge-m3`) must be loaded before documents can be ingested
- The reranker (`bge-reranker-v2-m3`) is optional but highly recommended — it dramatically improves retrieval quality

### systemd service restarts in a loop
```bash
journalctl -u llama-server.service --since "10 min ago" -f
```
Check paths in `ExecStart` and ensure the binary is executable.

---

## 📁 Repository Structure

```
local_llm/
├── .config/llamacpp/presets.ini   # Configurations for all 3 models (Qwen + BGE)
├── llama-server.service            # systemd service for background deployment
├── README.md                       # This file — full guide and documentation
└── CHANGELOG.md                    # Change history
```

---

## 🙏 Acknowledgements & Credits

- [llama.cpp](https://github.com/ggerganov/llama.cpp) by @ggerganov — the engine that makes all of this possible
- [@DogukanUrker](https://x.com/dogukanurker/status/2077472690156511643) — for proving Qwen3.6-35B-A3B runs on an RTX 3060 and sharing the config
- [Qwen](https://github.com/QwenLM/Qwen) by Alibaba Group — the open-weight model family
- [BGE Models](https://github.com/BAAI-Dada/BGE) — embeddings and reranking from BAAI

---

> 🏠 **No cloud. No API. Just consumer hardware doing something it has no business doing.**
