"""Trains policy with cma-es (minimal math, easy to code)"""
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm, trange

from src.mlp_actor import MLPActor
from src.simulate import rollout

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
 
        num_parents = len(sols_rewards//top_prop)

        evolution_operator.tell(ranking_indices,
                                ranking_values,
                                num_parents)
        print(max(sols_rewards))

actor_config = {
    "layer_shapes": [(4, 128),
                     (128, 128),
                     (128, 1)],
    "activation": nn.ReLU
}

actor = MLPActor(**actor_config)
params = actor.serialize()

train(params=params,
      actor_config=actor_config,
      num_iters=100)
