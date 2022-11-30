import os

import torch

class GlobalConfig:
    """ base architecture configurations """
    # refer to https://github.com/valeoai/LearningByCheating/blob/master/benchmark_agent.py
    path_folder_model = '../MaRLn_CARLA_challenge/model_RL_IAs_CARLA_Challenge/'

    nb_action_steering = 27
    max_steering = 0.6
    nb_action_throttle = 3
    max_throttle = 1.0

    front_camera_width = 288
    front_camera_height = 288
    front_camera_fov = 100
    crop_sky = False

    render = True
    disable_cuda = False
    # disable_cudnn = False

    # IQN parameters
    kappa = 1.0
    num_tau_samples = 8
    num_tau_prime_samples = 8
    num_quantile_samples = 32
    quantile_embedding_dim = 64

    steps_image = [
        -10,
        -2,
        -1,
        0,
    ]

    device = torch.device('cuda')

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
