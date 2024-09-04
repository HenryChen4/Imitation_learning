"""
MLP actor for continous control.
"""
import torch.nn as nn
import numpy as np
import torch

class MLPActor(nn.Module):
    def __init__(self,
                 layer_shapes,
                 activation):
        super().__init__()
        layers = []
        for i, shape in enumerate(layer_shapes):
            layers.append(
                nn.Linear(shape[0],
                          shape[1],
                          bias=shape[2] if len(shape) == 3 else True)
            )
            if i == len(layer_shapes) - 1:
                layers.append(nn.Tanh())
            else:
                layers.append(activation())
        self.model = nn.Sequential(*layers)

    def forward(self, obs):
        return self.model(obs)     

    def action(self, obs):
        obs = torch.from_numpy(obs[None].astype(np.float32))
        return self(obs)[0].cpu().detach().numpy()

    def initialize(self, func): 
        def init_weights(m):
            if isinstance(m, nn.Linear):
                func(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  
        self.apply(init_weights)

        return self

    def serialize(self):
        return np.concatenate(
            [p.data.cpu().detach().numpy().ravel() for p in self.parameters()])

    def deserialize(self, array):
        array = np.copy(array)
        arr_idx = 0
        for param in self.model.parameters():
            shape = tuple(param.data.shape)
            length = np.product(shape)
            block = array[arr_idx:arr_idx + length]
            if len(block) != length:
                print(len(block))
                print(length)

                raise ValueError("Array not long enough!")
            block = np.reshape(block, shape)
            arr_idx += length
            param.data = torch.from_numpy(block).float()
        return self