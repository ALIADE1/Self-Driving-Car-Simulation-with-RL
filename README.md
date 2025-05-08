# 🏎️ PPO Reinforcement Learning – CarRacing-v3 (Final Project)

This project is a reinforcement learning implementation using **Proximal Policy Optimization (PPO)** to train an agent in the `CarRacing-v3` environment from OpenAI Gym. The goal is to train an agent that learns to drive smoothly and efficiently on a procedurally generated track, with enhanced reward shaping and live visual rendering.

---

## 📂 Project Structure


---

## 🚀 Features

- **Stable Baselines3 PPO** with CNN policy.
- **Custom environment wrapper** for reward shaping and real-time rendering.
- **Console logger callback** that prints reward per episode.
- **Auto-stop** functionality after a timeout (5 minutes) to avoid endless training loops.
- **Seeded for reproducibility**.

---

## 🧠 Reward Shaping Logic

The wrapper:
- Encourages long-distance travel by rewarding the distance covered.
- Slightly boosts throttle and dampens braking for smoother driving.
- Applies small penalties for negative progress (wrong direction).
- Adds a continuous reward for staying on the track.

---

## 📦 Requirements

Install dependencies using:

```bash
pip install stable-baselines3[extra] gymnasium[box2d] torch numpy

python Code.py
