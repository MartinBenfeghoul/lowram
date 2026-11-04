# LowRAM
[![Website](https://img.shields.io/badge/Website-LowRAM-blue)](https://martinbenfeghoul.github.io/lowram/)

Supplementary code for **LowRAM: Low-Rank Keys and Amortised Values for Composable KV Cache Compression at Test-Time**.

LowRAM compresses the KV cache on two sides. Keys get a low-rank factorisation that is shared across groups of adjacent layers (xKV). Values are amortised: values are largely predictable from keys, so a small per-layer MLP `V̂ = f(K)` is fitted to each prompt at prefill. Only the tokens with the largest prediction error are kept as exact residuals. Evaluation runs through [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) on RULER and LongBench.

## How it works

- **`CompressedCache`** is a drop-in Transformers cache. You configure keys and values separately, and each backend reports its own byte count for compression-ratio accounting.
- **Key backends:** `baseline` (uncompressed), `xkv` (shared low-rank factors per layer group, with the rank set by the target ratio; the factors can optionally be quantised), and `turboquant` (random rotation + Lloyd-Max scalar quantisation).
- **Value backends:** `baseline`, `turboquant`, and `mlp`. With `mlp`, each layer fits an MLP to its KV pairs at prefill and sparsely stores the highest-error residuals that fit in the budget. The MLP has an optional linear branch, initialised from `pinv(W_k) @ W_v`.
- **Meta-learning:** first-order MAML over FineWeb-Edu learns the MLP initialisation, optimising the post-residual reconstruction loss.
- **Selective reconstruction:** landmark scoring at decode time rebuilds only the key chunks that matter (it uses xKV's fused CUDA kernels when they are available).
- **Attention-predictor eviction:** a small shared CNN predicts attention over key blocks. It drives eviction of prefill tokens and picks which value residuals to keep.

## Setup

```bash
conda create -n lowram python=3.12 -y
conda activate lowram
pip install -r requirements.txt
```

Create a `.env` in the repo root:

```
HF_TOKEN="<your_huggingface_token>"
```

Optional: build the fused CUDA kernels used by `--selective_reconstruction` up front. Otherwise they are JIT-built on first use, with a fallback to a torch path. This needs nvcc, ninja and an sm_80+ GPU.

```bash
python -m efficiency.install_kernel
```

## Checkpoints

The pretrained checkpoints are stored with [Git LFS](https://git-lfs.com). After cloning, fetch them with:

```bash
git lfs install
git lfs pull
```

Checkpoints are provided for `Llama-3.1-8B-Instruct` and `Mistral-7B-Instruct-v0.3`:

| Path | Flag |
| --- | --- |
| `checkpoints/meta_init/<model>/meta_mlps.pt` | `--meta_weights_path` |
| `checkpoints/value_mlps/<model>/value_mlps.pt` | `--value_mlp_weights_path` |
| `checkpoints/attention_predictor/<model>/model_ckpt.pt` | `--attn_predictor_path` (reads the `config.yaml` next to it) |

Use the checkpoint that matches the model you are evaluating.

## Usage

### Demo

The demo hides three access codes in a 4k-token document and asks Llama-3.1-8B-Instruct to retrieve them through a 4x-compressed LowRAM cache. It then prints the model's answer next to the reference codes. Pass `--method baseline` to compare against an uncompressed cache.

```bash
python -m examples.lowram_demo
```

### Evaluation

```bash
# uncompressed reference
python lm_eval_script.py --tasks ruler -kc baseline -vc baseline

# LowRAM: xKV keys + meta-learned MLP values at 4x
python lm_eval_script.py --tasks ruler \
    -kc xkv --comp_ratio 4.0 --xkv_layer_group_size 4 \
    -vc mlp --target_cr 4.0 \
    --num_epochs 2 \
    --meta_weights_path checkpoints/meta_init/Llama-3.1-8B-Instruct/meta_mlps.pt \
    --batch_size 4
```

| Flag | Effect |
| --- | --- |
| `-kc` / `-vc` | key / value cache backend |
| `-r/--comp_ratio` | target compression ratio for xKV keys |
| `--target_cr` | target compression ratio for MLP values |
| `--selective_reconstruction` | landmark-guided partial key reconstruction |
| `--eviction_keep_ratio` | fraction of prefill tokens kept (`<1` needs `--attn_predictor_path`) |
| `--limit` | samples per task |

Run `python lm_eval_script.py --help` for the full list.

### Training

```bash
# meta-learned MLP init (config in config/meta_learning.yaml)
python meta_learning.py training.inner_steps=15 training.eval_target_cr=4.0

# plain value-MLP pretraining
python train_value_mlps.py --model_name meta-llama/Llama-3.1-8B-Instruct \
    --seq_len 4096 --max_batches 32 --num_epochs 25

# attention predictor
python train_attention_predictor.py --model_name meta-llama/Llama-3.1-8B-Instruct \
    --seq_len 4096 --block_size 16 --history_step 64 --topk_blocks 16 --layers all
```
