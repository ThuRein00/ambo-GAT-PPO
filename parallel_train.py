import numpy as np
import gymnasium as gym
import torch
import DES_ambo as DES_ambo
from stable_baselines3 import PPO
import pandas as pd

from stable_baselines3.common.vec_env import SubprocVecEnv
from multiprocessing import cpu_count
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.monitor import Monitor
import pickle
import torch.multiprocessing as mp

from GAT import CustomGAT

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device}")

def linear_schedule(initial_learning_rate=5e-4, final_learning_rate=5e-6):
    """
    Decay linearly for first half, hold constant at final_lr for second half
    """
    def func(progress_remaining):
        progress = 1 - progress_remaining  
        
        if progress < 0.5:
            # first half (linear decay from initial to final)
            decay_progress = progress / 0.5  
            lr = initial_learning_rate + decay_progress * (final_learning_rate - initial_learning_rate)
        else:
            # second half (constant at final_lr)
            lr = final_learning_rate
        
        return lr
    return func


def make_env(env_id: str, rank: int, seed: int = 0, env_kwargs: dict = {}):
    """
    Factory function for multiprocessing
    """
    def _init():
        env = gym.make(env_id, **env_kwargs)
        # use a different seed for each env
        env.reset(seed=seed + rank)
        return Monitor(env)
    return _init


if __name__ == "__main__":
    
    
    # Use 'spawn' instead of the default 'fork' to start subprocesses.
    # Each subprocess starts as a fresh Python interpreter and builds its own env cleanly.
    # Required to avoid CUDA/PyTorch crashes that occur when forking a process with GPU state.
    mp.set_start_method('spawn', force=True)
    
    
    #Data Loading 
    file_paths = {
        "accident_rate": "data/accident_rate.csv",
        "distance_Base_to_Incident_df": "data/distance_base_to_incident.csv",
        "distance_Hospital_to_Base_df": "data/distance_hospital_to_base.csv",
        "nearest_place" : "data/nearest_places_data.csv",
        "ambulance_initialization" : "data/ambulance_initialization.csv"
    }
    # Load data
    print("Loading data...")
    accident_rate = pd.read_csv(file_paths["accident_rate"])
    distance_Base_to_Incident_df = pd.read_csv(file_paths["distance_Base_to_Incident_df"])
    distance_Hospital_to_Base_df = pd.read_csv(file_paths["distance_Hospital_to_Base_df"])
    nearest_place = pd.read_csv(file_paths["nearest_place"])
    ambulance_initialization = pd.read_csv(file_paths["ambulance_initialization"])
    ambulance_initialization_dict = ambulance_initialization.to_dict()['initial_ambulances']
    with open('data/incident_pred.pkl', 'rb') as f:
        accident_rate_pred = pickle.load(f)
    print(np.array(accident_rate_pred).shape)
    print("Data loading complete.")

    # Env Set up
    print("Creating environment...")
    env_id = "DES_ambo/DES_ambo_map-train"
    env_kwargs = dict(
        accident_rate = accident_rate,
        accident_rate_pred = accident_rate_pred,
        distance_Base_to_Incident_df = distance_Base_to_Incident_df,
        distance_Hospital_to_Base_df = distance_Hospital_to_Base_df,
        nearest_place=nearest_place,
        init_ambulances_per_base_dict=ambulance_initialization_dict,
        run_until=1440,
        trace=False,
        test = False,
        flat_feature = False,
    )
    
    num_cpu = cpu_count()
    print(f"Using {num_cpu} parallel CPU processes for environment simulation.")
    
    # Create the SubprocVecEnv 
    env = SubprocVecEnv([make_env(env_id, i, env_kwargs=env_kwargs) for i in range(num_cpu)])
    
    MODEL_NAME = "with_decision"
    MODEL_SAVE_PATH = f"model/{MODEL_NAME}"
    STATS_SAVE_PATH = f"model/{MODEL_NAME}_stat.pkl"

    norm_env = VecNormalize(env, norm_obs=False, norm_reward=True)


    model = PPO(
    "MlpPolicy",
    norm_env,
    learning_rate=linear_schedule(initial_learning_rate=5e-4,final_learning_rate=1e-5),
    n_steps=1024,
    batch_size=512,
    n_epochs=3,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0,
    vf_coef=1.0,
    max_grad_norm=0.5,
    policy_kwargs=dict(
        features_extractor_class=CustomGAT,
        features_extractor_kwargs=dict(features_dim=1024+52*64),
        net_arch=[1024,512]
    ),
    verbose=1,
    device=device,
    tensorboard_log="./GAT_PPO/"
)
    print("Starting training for 35M steps...")
    model.learn(
        total_timesteps=20_000_000,
        tb_log_name=MODEL_NAME,
    )

    print("Training complete.")
    model.save(MODEL_SAVE_PATH)
    norm_env.save(STATS_SAVE_PATH)