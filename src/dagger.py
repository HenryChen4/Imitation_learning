import torch.nn as nn
import torch
import gymnasium as gym
import numpy as np

from tqdm import trange

from mlp_actor import MLPActor
from simulate import rollout

def DAgger(imitator_config,
           num_iters,
           demonstrator,
           beta=None):
    """ Implementation of DAgger algorithm.
    Args:
        imitation_config (dict): Config for imitation policy.
        dc_config (dict): Config for data collection policy.
        num_iters (int): Number of training iterations.
        demonstrator (MLPActor): Demonstrator MODEL.
        beta (float): Discount.
    """

    dataset = []
    imitator = MLPActor(**imitator_config)
    imitator.initialize(nn.init.zeros_)
    
    for i in trange(num_iters):
        if beta == None:
            beta = i/num_iters

        demonstrator_params = demonstrator.serialize()
        imitator_params = imitator.serialize()
        new_imitator_params = beta * (demonstrator_params) + (1 - beta) * imitator_params

        trajectory, rewards = rollout(actor_config=imitator_config,
                                      params=new_imitator_params)
        
        
