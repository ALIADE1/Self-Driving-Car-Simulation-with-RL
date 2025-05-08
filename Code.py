#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPO RL Final Project (Improved)
Live training window for CarRacing-v3
"""

import os
import random
import time
import numpy as np
import torch
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# ─── 1) Seed for reproducibility ──────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


# ─── 2) Reward-shaping + auto-render wrapper ──────────────────
class RenderAndRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.total_distance = 0.0
        self.frame_skip = (
            1  #  Make the agent observe every frame for more accurate learning
        )

    def reset(self, **kwargs):
        self.total_distance = 0.0
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        # Gradual adjustment to the controls
        boosted_action = np.array(
            [
                action[0],
                np.clip(action[1] + 0.2, 0, 1),  # Gradual increase in acceleration
                np.clip(action[2] - 0.1, 0, 1),  # Gradual decrease in braking
            ],
            dtype=np.float32,
        )

        total_reward = 0
        terminated = False
        truncated = False
        info = {}

        for _ in range(self.frame_skip):
            obs, reward, term, trunc, info = self.env.step(boosted_action)
            total_reward += reward
            terminated = terminated or term
            truncated = truncated or trunc
            if terminated or truncated:
                break

        # Reward shaping based on the distance traveled
        if total_reward > 0:
            self.total_distance += (
                0.01  # Slight increase in distance when moving in the correct direction
            )
            total_reward += (
                0.5 * self.total_distance
            )  # Encourage more for longer distances
        else:
            total_reward -= 0.5  # Light punishment for going the wrong way

        total_reward += (
            0.001 * self.total_distance
        )  # Additional reward for staying on the track
        self.env.render()
        return obs, total_reward, terminated, truncated, info


# ─── 3) Console logger callback ───────────────────────────────
class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.ep_rewards = []
        self.start_time = time.time()

    def _on_step(self) -> bool:
        self.ep_rewards.append(self.locals["rewards"][0])
        if self.locals["dones"][0]:
            print(f"Episode done | Total reward: {sum(self.ep_rewards):.2f}")
            self.ep_rewards = []
        # Safety check: auto-stop after 5 minutes
        if time.time() - self.start_time > 300:  #  5 minutes for longer training time
            print("🛑 Auto-stopping training (timeout reached)")
            return False
        return True


# ─── 4) Create and wrap the environment ───────────────────────
env = gym.make("CarRacing-v3", render_mode="human")
env = RenderAndRewardWrapper(env)
env.reset(seed=SEED)

# ─── 5) Create the PPO model (Improved Settings) ───────────────
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,  #  Slower learning rate for higher accuracy
    n_steps=2048,  #  More steps before update
    batch_size=128,  #  Larger batch size for more stability
    gamma=0.999,  #  Focus more on long-term rewards
    gae_lambda=0.98,  #  Generalized Advantage Estimation improvement
    clip_range=0.2,
    seed=SEED,
)

# ─── 6) Train with live window ────────────────────────────────
TIMESTEPS = 2000  #  Longer training for improved performance
print(f"Training for {TIMESTEPS} timesteps (live window)...")
model.learn(total_timesteps=TIMESTEPS, callback=RewardLogger())

# ─── 7) Save model ────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
model.save("models/ppo_carracing_live")
print("✅ Model saved to models/ppo_carracing_live.zip")

# ─── 8) Close environment ─────────────────────────────────────
env.close()
print("🏁 Training finished successfully!")
