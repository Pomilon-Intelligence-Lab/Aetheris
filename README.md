# Aetheris: Hybrid Mamba-MoE Experiment

<p align="center">
  <img src="https://img.shields.io/badge/Status-Experimental-yellow.svg" alt="Status">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-orange.svg" alt="PyTorch">
</p>



**Aetheris** is a hobbyist research project and experimental implementation exploring the intersection of **State Space Models (Mamba)** and **Mixture of Experts (MoE)**.

The goal of this project was to learn by doing: attempting to combine the linear-time inference of Mamba with the sparse scaling capacity of MoE from scratch in PyTorch. It is designed as a playground for understanding these modern architectures, not as a published academic paper or production-ready foundation model.

## 🧪 The Experiment

Current LLM architectures are evolving rapidly. I built Aetheris to investigate a specific question:
> *Can we successfully interleave Mamba blocks (for long context) with sparse MoE layers (for capacity) to train an efficient model on consumer hardware?*

This project implements a hybrid architecture that attempts to:
1.  **Replace Attention:** Use Mamba (SSM) blocks to achieve $O(N)$ sequence scaling.
2.  **Scale Parameters Sparsely:** Use MoE layers to increase model size without exploding the computational cost per token.
3.  **Run Locally:** Optimize the implementation for single-GPU training (gradient checkpointing, efficient routing).

## 🏗️ Architecture Implementation

Aetheris alternates between custom implementations of two core modules:

* **SSMBlock (The Backbone):** Implements the selective scan mechanism described in the [Mamba paper](https://arxiv.org/abs/2312.00752). This handles the sequence mixing and "memory" of the model.
* **SparseMoELayer (The Scaling):** A router-based layer that dispatches tokens to Top-K experts (Feed-Forward Networks). This allows the model to "specialize" parts of its parameters for different types of tokens.

## 🚀 Quick Start

This code is provided for educational purposes and for others who want to experiment with hybrid architectures.

### Installation

```bash
git clone https://github.com/Pomilon/Aetheris.git
cd Aetheris
pip install -r requirements.txt
````

### Usage (CLI)

Aetheris includes a CLI to train or inference the model.

**1. Training (From Scratch)**

```bash
# Trains a small model defined in configs/default.yaml
python -m aetheris.cli.main train --config configs/default.yaml
```

**2. Generation**

```bash
python -m aetheris.cli.main generate --prompt "The quick brown fox" --checkpoint_dir checkpoints
```

## ⚙️ Configuration

You can tweak the hyperparameters in `configs/`. I've included a "Debug" config that is small enough to train on a laptop CPU for testing the code flow.

| Config File | Description |
| :--- | :--- |
| `configs/default.yaml` | Standard experimental setup (requires GPU). |
| `configs/debug.yaml` | Tiny model (2 layers) for code debugging. |

## 📚 Acknowledgements & References

This project is an implementation study and relies heavily on the brilliant theoretical work of others. It is not an original invention of the Mamba or MoE concepts.

  * **Mamba Architecture:** Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
  * **Mixture of Experts:** Shazeer, N., et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. [arXiv:1701.06538](https://arxiv.org/abs/1701.06538)
  * **Inspiration:** Jamba (AI21 Labs) and OpenMoE.

## 🧠 Model Weights & Checkpoints

All pre-trained checkpoints are hosted on the [Hugging Face Hub](https://huggingface.co/Pomilon).

| Model Artifact | Step | Description | Download |
| :--- | :--- | :--- | :--- |
| **Aetheris-Base** | 10k | Early convergence checkpoint (Loss ~3.66). Good for analyzing router behavior. | [🤗 Hugging Face](https://huggingface.co/Pomilon/Aetheris) |
| **Aetheris-Chat** | -- | *Coming Soon (Post-SFT)* | -- |

> **⚠️ Important:** Aetheris uses a custom Hybrid Mamba-MoE architecture. You **cannot** load it directly with `transformers.AutoModel`. You must use the interface provided in this repository.

### 🐍 How to Load

```python
python -m aetheris.cli.main generate --prompt "The quick brown fox" --checkpoint_dir path/to/checkpoints_folder # rename the checkpoint inside to checkpoint_current.pth
```
> **Note:** will add better inference later down the line, for now used this scuffed version. :D

> **Note:** These weights are from an experimental run. While they demonstrate the architectural capabilities, do not expect GPT-5 or even google bard level coherence. :D
> this project was made for learning and fun!

## License

MIT
