<div align="center">

# Looped Diffusion Language Models

**Iterative refinement in masked diffusion language models through shared Transformer layers.**

[Getting Started](#getting-started) | [Training](#training) | [Baselines](#baselines) | [Checkpoints](#checkpoints) | [Acknowledgements](#acknowledgements)

</div>

This repository contains the language-model training code for **LoopMDM**.
The denoising Transformer consists of a head, a shared middle block applied
multiple times, and a tail. During training, the loop count is sampled uniformly
from **1 to 12**, with gradients propagated through all iterations.

The repository includes:

- **LoopMDM** training on LM1B and OpenWebText.
- **MDLM** baseline training with the same dataset and optimizer settings.
- EMA, gradient accumulation, and checkpoint saving and resuming.

# Getting Started

Create an environment and install the dependencies. Use a CUDA development
environment compatible with PyTorch 2.2.1 to build FlashAttention.

```bash
conda create -n loopmdm python=3.9
conda activate loopmdm

pip install -r requirements.txt
pip install flash-attn==2.5.6 --no-build-isolation
```

Run the commands below from the repository root. Datasets are downloaded and
preprocessed on the first training run. To choose a dataset-cache location, set:

```bash
export DATA_CACHE_DIR=/absolute/path/to/dataset_cache
```

The default cache location is `data_cache/` under the launch directory.

# Training

Use [`scripts/run.py`](scripts/run.py) to train LoopMDM. All training commands
default to **1M optimizer steps** on **4 GPUs**, with an effective global batch
size of **512**.

## LM1B

LM1B uses the `bert-base-uncased` tokenizer with sentence packing and sequence
length 128.

```bash
python scripts/run.py train \
  --dataset lm1b \
  --model loop \
  --output outputs/lm1b-loop
```

## OpenWebText

OpenWebText uses the GPT-2 tokenizer and packed sequences of length 1024.
The last 100,000 documents are excluded from the training split.

```bash
python scripts/run.py train \
  --dataset owt \
  --model loop \
  --output outputs/owt-loop
```

## Configuration

Default settings are defined in [`configs/config.yaml`](configs/config.yaml)
and [`configs/model/small.yaml`](configs/model/small.yaml).

| Setting | Default |
| --- | --- |
| Transformer layers | 12 |
| Hidden size / attention heads | 768 / 12 |
| Head / looped middle / tail layers | 1 / 2 / 9 |
| Training loop count | Uniformly sampled from 1–12 |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Warmup | 2,500 steps |
| EMA decay | 0.9999 |
| Batch size per GPU | 8 |
| Global batch size | 512, using gradient accumulation |
| Training steps | 1,000,000 |

**Notes:**

- Set `--devices` and `--batch-size` to configure GPU count and batch size per
  GPU. Their product must divide 512.
- Set `--max-steps` to change the training duration.
- Set `--head`, `--tail`, and `--max-loop` to configure the looped block.
- Use `--override KEY=VALUE` for additional Hydra settings. The flag can be repeated.
- Use a separate `--output` directory for each run. W&B is disabled by the launcher.

To inspect a command without starting training:

```bash
python scripts/run.py train --dataset owt --model loop --dry-run
```

# Baselines

To train the MDLM baseline, use `--model baseline`:

```bash
python scripts/run.py train \
  --dataset owt \
  --model baseline \
  --output outputs/owt-baseline
```

Use `--dataset lm1b` to train on LM1B. Baselines
also train for **1M optimizer steps** by default.

# Checkpoints

Checkpoints are saved every **50,000 optimizer steps** in the run's
`checkpoints/` directory. `last.ckpt` tracks the latest saved checkpoint.
Checkpoints include model parameters, EMA weights, optimizer state, and
training progress.

To resume training, use the same model and dataset settings and supply the
checkpoint path:

```bash
python scripts/run.py train \
  --dataset owt \
  --model loop \
  --output outputs/owt-loop \
  --override checkpointing.resume_from_ckpt=true \
  --override checkpointing.resume_ckpt_path=/absolute/path/last.ckpt
```

# Acknowledgements

This code builds on [MDLM](https://github.com/kuleshov-group/mdlm).

# License

This repository is distributed under the Apache-2.0 license. See [LICENSE](LICENSE).
