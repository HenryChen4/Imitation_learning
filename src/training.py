"""Trains policy with cma-es (minimal math, easy to code)"""
import torch.nn as nn
import torch
import numpy as np

from src.mlp_actor import MLPActor
from src.simulate import rollout

from ribs.emitters.opt import CMAEvolutionStrategy

def train(params,
          actor_config,
          num_iters,
          sigma0=0.1,
          batch_size=30,
          seed=135135):
    # initialize evolution strategy operator
    solution_dim = params.shape[0]
    evolution_operator = CMAEvolutionStrategy(sigma0=sigma0,
                                              batch_size=batch_size,
                                              solution_dim=solution_dim,
                                              seed=seed)

    print(solution_dim)

    initial_mean = np.mean(params) 
    evolution_operator.reset(initial_mean)
    
    for i in range(num_iters):
        # sample new parameters and give them to model
        sols = evolution_operator.ask()

        # TODO: Use DASK eventually for multithreading
        for sol_i in sols:
            actor = MLPActor(**actor_config)
            actor.deserialize(sols)

            # retrieve rewards of actor
            _, rewards = rollout(actor_config=actor_config,
                                 params=sols)
            print(rewards)

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
