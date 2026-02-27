# TP-GNN-PPO: Intelligent Orchestration of Trusted Service Function Chains

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-EE4C2C.svg)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/Paper-IEEE_Conference-green)]() This is the official PyTorch implementation for the paper **"TP-GNN-PPO: Intelligent Orchestration of Trusted Service Function Chains via Topology-Aware Reinforcement Learning"**.

## 📖 Overview
The paradigm shift towards Zero-Trust Architecture (ZTA) in modern 5G/6G networks imposes stringent security constraints on Service Function Chain (SFC) provisioning. This repository provides **TP-GNN-PPO**, a topology-aware and security-deterministic Deep Reinforcement Learning (DRL) framework tailored for zero-trust environments.

### Core Innovations
1. **Topology-Aware GNN Encoder (`models/tp_gnn.py`)**: Extracts global state representations to mitigate resource exhaustion and avoid congested bottlenecks.
2. **Deterministic Trust Action Masking (`models/ppo_agent.py`)**: Strictly enforces ZTA compliance at the action-space level, filtering out unsafe nodes before sampling to eliminate reward hacking.
3. **Adaptive Curriculum Learning (`mainV3.py`)**: Gradually intensifies the trust threshold $\tau$ during training, ensuring stable exploration and mitigating sparse-reward issues.

---

## 🏗️ Architecture
![TP-GNN-PPO Architecture](assets/architecture.png)
*(Please see the paper for detailed mathematical formulations).*

---

## ⚙️ Installation

1. **Clone this repository:**
```bash
git clone [https://github.com/YourUsername/TP-GNN-PPO.git](https://github.com/YourUsername/TP-GNN-PPO.git)
cd TP-GNN-PPO