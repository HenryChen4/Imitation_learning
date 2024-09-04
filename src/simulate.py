"""
Single rollout of unbatched policy for inverted pendulum task
"""
import numpy as np
import gymnasium as gym
import torch

from src.mlp_actor import MLPActor

def rollout(actor_config,
            params=None,
            seed=None):
    env = gym.make("InvertedPendulum-v4")

    action_dim = np.prod(env.action_space.shape)
    obs_dim = env.observation_space.shape[0]
    episode_length = env.spec.max_episode_steps

    total_reward = 0.0

    obs, _ = env.reset(seed=seed)

    trajectories = {
        "state": np.full((episode_length, obs_dim), np.nan, dtype=np.float32),
        "action": np.full((episode_length, action_dim), np.nan, dtype=np.float32),
        "reward": np.full((episode_length, ), np.nan, dtype=np.float32),
        "next_state": np.full((episode_length, obs_dim), np.nan, dtype=np.float32)
    }

    model = MLPActor(**actor_config)
    if params is not None:
        model.deserialize(params)

    total_timesteps = 0
    done = False
    while not done:
        action = model.action(obs)
        old_obs = obs
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        trajectories["state"][total_timesteps] = old_obs
        trajectories["action"][total_timesteps] = action
        trajectories["reward"][total_timesteps] = reward
        trajectories["next_state"][total_timesteps] = obs

        total_timesteps += 1

    return trajectories, total_reward
