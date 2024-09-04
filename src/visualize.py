import torch.nn as nn
import gymnasium as gym
import numpy as np
import torch
import glob
import io
import base64

from IPython.display import display, HTML
from src.simulate import rollout

def load_model(model_save_path):
    model = torch.load(model_save_path, map_location=torch.device("cpu"))
    all_np_params = []
    for param in model.parameters():
        np_params = param.data.numpy()
        if len(np_params.shape) == 2:
            reshaped_params = np_params.reshape(np_params.shape[0] * np_params.shape[1])
            all_np_params.append(reshaped_params)
        else:
            all_np_params.append(np_params)
    return_params = []
    for subparams in all_np_params:
        return_params.extend(subparams)
    return return_params

def visualize_trajectory(model_params,
                         actor_config):
    video_dir = "./results/expert_videos"

    env = gym.make("InvertedDoublePendulum-v5", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_dir)
    
    rollout(actor_config=actor_config,
            params=model_params,
            video_env=env)
    
    env.close()

    for video_file in glob.glob(f"{video_dir}/*.mp4"):
        video = io.open(video_file, 'rb').read()
        encoded = base64.b64encode(video).decode("ascii")
        display(
            HTML(f'''
            <video width="360" height="auto" controls>
                <source src="data:video/mp4;base64,{encoded}" type="video/mp4" />
            </video>'''))

model_params = load_model("./models/models/model.pth")

actor_config = {
    "layer_shapes": [(8, 128),
                     (128, 128),
                     (128, 1)],
    "activation": nn.ReLU
}

visualize_trajectory(model_params,
                     actor_config)