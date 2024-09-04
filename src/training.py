import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch
import os

from src.mlp_actor import MLPActor
from src.simulate import rollout

from tqdm import tqdm, trange
from ribs.emitters.opt import CMAEvolutionStrategy

def train(params,
          actor_config,
          num_iters,
          top_prop=4,
          sigma0=0.1,
          batch_size=30,
          seed=135135):
    solution_dim = params.shape[0]
    evolution_operator = CMAEvolutionStrategy(sigma0=sigma0,
                                              batch_size=batch_size,
                                              solution_dim=solution_dim,
                                              seed=seed)
    initial_mean = np.mean(params) 
    evolution_operator.reset(initial_mean)
    
    print("> Beginning black box search")
    best_rewards = []
    best_solutions = []
    num_max = 0
    for i in trange(num_iters):
        sols = evolution_operator.ask()

        # TODO: Use DASK for multithreading
        sols_rewards = []
        for sol_i in sols:
            actor = MLPActor(**actor_config)
            actor.deserialize(sol_i)
            _, sol_i_reward = rollout(actor_config=actor_config,
                                      params=sol_i)
            sols_rewards.append(sol_i_reward)
        
        ranking_indices = np.flip(np.argsort(sols_rewards))
        ranking_values = sols_rewards
 
        num_parents = len(sols_rewards)//top_prop

        evolution_operator.tell(ranking_indices,
                                ranking_values,
                                num_parents)

        print(f"iter {i} reward: {max(sols_rewards)}")

        best_rewards.append(max(sols_rewards))
        np_best_rewards = np.array(sols_rewards)
        best_solution = sols[np.argmax(np_best_rewards)]
        best_solutions.append(best_solution)

        if max(sols_rewards) == 1000:
            num_max += 1

        if num_max > 3:
            break;    

    return best_rewards, best_solutions[-1]

# TODO: Refactor to main
actor_config = {
    "layer_shapes": [(8, 128),
                     (128, 128),
                     (128, 1)],
    "activation": nn.ReLU
}

actor = MLPActor(**actor_config)
params = actor.serialize()

best_rewards, best_solution = train(params=params,
                                    actor_config=actor_config,
                                    num_iters=100)

# save rewards plot
total_iters = np.arange(len(best_rewards))
os.makedirs("./models", exist_ok=True)
reward_save_path = os.path.join("./models", "best_rewards.png")
plt.plot(total_iters, best_rewards)
plt.xlabel("Iteration")
plt.ylabel("Best reward")
plt.savefig(reward_save_path)

# save model
model_save_path = os.path.join("./models", "model.pth")
actor.deserialize(best_solution)
torch.save(actor, model_save_path)