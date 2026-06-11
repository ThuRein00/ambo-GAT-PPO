import numpy as np
import pandas as pd
import gymnasium as gym
import DES_ambo as DES_ambo
import pickle
import torch as th
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import pickle
from torch_geometric.utils import to_undirected


#Data Loading 
file_paths = {
        "accident_rate": "final_version/data/accident_rate.csv",
        "distance_Base_to_Incident_df": "final_version/data/distance_base_to_incident.csv",
        "distance_Hospital_to_Base_df": "final_version/data/distance_hospital_to_base.csv",
        "distance_Base_to_Base_df": "final_version/data/distance_base_to_base.csv",
        "nearest_place" : "final_version/data/nearest_places_data.csv",
        "ambulance_initialization" : "final_version/data/ambulance_initialization.csv"
    }
    # Load data
print("Loading data...")
accident_rate = pd.read_csv(file_paths["accident_rate"])
accident_rate = accident_rate /2
accident_rate.to_csv("data/accident_rate")
print(f"input {accident_rate['mean_rate'].sum()}")
distance_Base_to_Incident_df = pd.read_csv(file_paths["distance_Base_to_Incident_df"])
distance_Hospital_to_Base_df = pd.read_csv(file_paths["distance_Hospital_to_Base_df"])
distance_Base_to_Base_df = pd.read_csv(file_paths["distance_Base_to_Base_df"])
nearest_place = pd.read_csv(file_paths["nearest_place"])
ambulance_initialization = pd.read_csv(file_paths["ambulance_initialization"])
ambulance_initialization_dict = ambulance_initialization.to_dict()['initial_ambulances']
with open('incident_pred.pkl', 'rb') as f:
    accident_rate_pred = pickle.load(f)
# print(np.array(accident_rate_pred).shape)
# print(np.array(accident_rate_pred))
# print("Data loading complete.")


