import torch.nn as nn
import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import time

import torch.utils
from tqdm import trange, tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from src.mlp_actor import MLPActor
from src.simulate import rollout
from src.visualize import load_model

class State_Action_Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

def create_loader(data, batch_size, shuffle=False):
    all_states = []
    all_actions = []
    for expert_trajectory in data:
        imitator_states = expert_trajectory["state"]
        expert_actions = expert_trajectory["action"]

        for imitator_state in imitator_states:
            if np.isnan(imitator_state).any():
                break
            all_states.append(imitator_state)
        
        for expert_action in expert_actions:
            if np.isnan(expert_action).any():
                break
            all_actions.append(expert_action)

    dataset = State_Action_Dataset(all_states, all_actions)
    return DataLoader(dataset, batch_size, shuffle)

def demonstrator_rollout(demonstrator,
                         visited_states):
    """ Rollout of demonstrator with ready states to collect
    demonstrator actions. 

    Args:
        demonstrator (MLPActor): Demonstrator model.
        visited_states (np.ndarray): States visited by imitator.
    Returns:
        demonstrator_response (dict): Dictionary containing states
            and corresponding actions by the demonstrator.
    """
    env = gym.make("InvertedPendulum-v4")

    action_dim = np.prod(env.action_space.shape)
    obs_dim = env.observation_space.shape[0]
    episode_length = env.spec.max_episode_steps

    demonstrator_response = {
        "state": np.full((episode_length, obs_dim), np.nan, dtype=np.float32),
        "action": np.full((episode_length, action_dim), np.nan, dtype=np.float32)
    }

    timestep = 0
    for obs in visited_states:
        action = demonstrator.action(obs)
        demonstrator_response["state"][timestep] = obs
        demonstrator_response["action"][timestep] = action
        timestep += 1

    return demonstrator_response

def train_imitator(imitator,
                   train_loader,
                   num_iters,
                   learning_rate):
    """ON GPU"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    imitator.to(device)

    optimizer = Adam(params=imitator.parameters(),
                     lr=learning_rate)
    criterion = torch.nn.MSELoss(reduction='mean')

    all_epoch_loss = []
    for epoch in trange(num_iters):
        epoch_loss = 0.0
        for i, (data_tuple) in enumerate(train_loader):
            obs = data_tuple[0].to(device)
            target_actions = data_tuple[-1]
            predicted_actions = imitator(obs)

            batch_loss = criterion(predicted_actions, target_actions.to(device))
            epoch_loss += batch_loss.item()

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        all_epoch_loss.append(epoch_loss/len(train_loader))

    return all_epoch_loss

def dagger(imitator_config,
           demonstrator,
           num_dagger_iters,
           train_batch_size,
           num_training_iters,
           learning_rate=1e-3,
           beta=None):
    """ Implementation of DAgger algorithm. ON CPU
    Args:
        imitation_config (dict): Config for imitation policy.
        dc_config (dict): Config for data collection policy.
        num_iters (int): Number of training iterations.
        demonstrator (MLPActor): Demonstrator MODEL.
        beta (float): Discount.
    """

    dataset = []
    imitator = MLPActor(**imitator_config)
    imitator.initialize(nn.init.kaiming_normal_)

    imitator_training_loss = []
    imitator_rewards = []
    iteration_times = []

    for i in trange(num_dagger_iters):
        if beta == None:
            beta = i/num_iters

        demonstrator_params = demonstrator.serialize()
        imitator_params = imitator.serialize()
        new_imitator_params = beta * (demonstrator_params) + (1 - beta) * imitator_params

        print(f"iter {i} params: {imitator_params}")

        start_time = time.time()

        trajectory, rewards = rollout(actor_config=imitator_config,
                                      params=new_imitator_params)
        imitator_states = trajectory["state"]

        demonstrator_response = demonstrator_rollout(demonstrator=demonstrator,
                                                     visited_states=imitator_states)
        
        dataset.append(demonstrator_response)   
        train_loader = create_loader(dataset, train_batch_size)

        training_loss = train_imitator(imitator,
                                       train_loader,
                                       num_training_iters,
                                       learning_rate)
        mean_training_loss = np.mean(training_loss)

        print(f"dagger iter {i}, training loss {mean_training_loss}, max reward {rewards}")

        imitator_training_loss.append(mean_training_loss)
        imitator_rewards.append(rewards)

        end_time = time.time()
        iteration_time = start_time - end_time
        iteration_times.append(iteration_time)

    return imitator_training_loss, imitator_rewards, iteration_times, imitator

# hyperparam init
imitator_config = {
    "layer_shapes": [(4, 128),
                     (128, 128),
                     (128, 1)],
    "activation": nn.ReLU
}

num_iters = 10

demonstrator_params = load_model("./old_models/passive/model.pth")
demonstrator_config = {
    "layer_shapes": [(4, 128),
                     (128, 128),
                     (128, 1)],
    "activation": nn.ReLU
}
demonstrator = MLPActor(**demonstrator_config)
demonstrator.deserialize(demonstrator_params)


# run dagger
imitator_training_loss, imitator_rewards, iteration_times, trained_imitator = dagger(imitator_config=imitator_config,
                                                                                     demonstrator=demonstrator,
                                                                                     num_dagger_iters=num_iters,
                                                                                     train_batch_size=2,
                                                                                     num_training_iters=100)

# plotting
fig, axs = plt.subplots(2, 1, figsize=(10, 8))
axs[0].plot(imitator_training_loss, label="Imitator training loss", color='blue')
axs[0].set_title('Imitator training loss over epochs')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].legend()

axs[1].plot(imitator_rewards, label='Imitator rewards', color='orange')
axs[1].set_title('Imitator rewards over epochs')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Rewards')
axs[1].legend()

plt.tight_layout()
os.makedirs("./results", exist_ok=True)
plot_save_path = os.path.join("./results", "loss_reward.png")
plt.savefig(plot_save_path)

plt.clf()

plt.plot(np.arange(len(iteration_times)), iteration_times)
plt.xlabel("Iteration")
plt.ylabel("Time taken")
time_save_path = os.path.join("./results", "time.png")
plt.savefig(time_save_path)

# save the trained imitator model
model_save_path = os.path.join("./models", "model.pth")
torch.save(trained_imitator, model_save_path)