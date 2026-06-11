import numpy as np
import pandas as pd
import gymnasium as gym
import DES_ambo as DES_ambo
import pickle
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

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
distance_Base_to_Incident_df = pd.read_csv(file_paths["distance_Base_to_Incident_df"])
distance_Hospital_to_Base_df = pd.read_csv(file_paths["distance_Hospital_to_Base_df"])
distance_Base_to_Base_df = pd.read_csv(file_paths["distance_Base_to_Base_df"])
nearest_place = pd.read_csv(file_paths["nearest_place"])
ambulance_initialization = pd.read_csv(file_paths["ambulance_initialization"])
ambulance_initialization_dict = ambulance_initialization.to_dict()['initial_ambulances']
with open('incident_pred.pkl', 'rb') as f:
    accident_rate_pred = pickle.load(f)
print(np.array(accident_rate_pred).shape)
print("Data loading complete.")


# Env Set up
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
        test = False
    )

env  = gym.make(
    env_id,
    **env_kwargs
    )

print("Environment Created...")


# sample 1 instance
obs, info = env.reset()
print("observation", obs)
print("shape" , obs.shape)



# # check obs dim
# for keys, values in obs.items():
#     print(f"Key {keys} : Shape {values.shape}")


# # Graph Construction 
# # nodes
# x = torch.tensor(np.stack([obs["ambo_count"] ,obs["relocation_travel_times"]],axis =1), dtype=torch.float)

# # fully connection
# # edges
# connection = []
# NUM_AMBO_BASES = distance_Base_to_Base_df.shape[0]
# for i in range(NUM_AMBO_BASES):
#     for j in range(NUM_AMBO_BASES):
#         connection.append([i,j])

# edge_index = torch.tensor((connection), dtype=torch.long)
# print(f"size of edge index {edge_index.size()}")

# # edge_attribute
# attributes = []
# for row in range(NUM_AMBO_BASES):
#     for column in range(NUM_AMBO_BASES):
#         if column < row:
#             attributes.append(distance_Base_to_Base_df.loc[row, f"base{column}"])
#         else:
#             attributes.append(distance_Base_to_Base_df.loc[column, f"base{row}"])


# edge_att = torch.tensor(np.array(attributes).reshape(-1,1),dtype=torch.float)
# print(f"size of edge attribute {edge_att.size()}")


# # Final Graph
# graph_data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_att)

# print(f"Number of nodes {graph_data.num_nodes}")
# print(f"Number of edges {graph_data.num_edges}")
# print(f"Number of node_features {graph_data.num_node_features}")

# # have self loops
# print(f"Is there self loops? {graph_data.has_self_loops()}")

# # have direction
# print(f"Is directed? {graph_data.is_directed()}")

# GAT1 = GATConv(in_channels = 2,
#               out_channels = 16, 
#               heads = 1,  
#               add_self_loops = False, 
#               edge_dim = 1, 
#               bias = True)

# GAT2 = GATConv(in_channels = 16,
#               out_channels = 16, 
#               heads = 1,  
#               add_self_loops = False, 
#               edge_dim = 1, 
#               bias = True)


# x_1 = GAT1.forward(x = graph_data.x,
#                       edge_index = graph_data.edge_index,
#                       edge_attr = graph_data.edge_attr)
# x_2 = GAT2.forward(x = x_1,
#                       edge_index = graph_data.edge_index,
#                       edge_attr = graph_data.edge_attr)

# x_2_flat = x_2.reshape(-1,1)
# print(x_2_flat.size())
