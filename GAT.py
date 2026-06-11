import torch as th
from gymnasium import spaces
from torch_geometric.data import Data
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_undirected
import numpy as np
import pandas as pd
import torch.nn.functional as F

distance_Base_to_Base_df = pd.read_csv("data/distance_base_to_base.csv")
NUM_AMBO_BASES = distance_Base_to_Base_df.shape[0]


# base 
# Build edge_index and edge_attribute from manual adjacency
adjacency = {
    0:  [1, 5],
    1:  [0, 2, 6],
    2:  [1, 7],
    3:  [4],
    4:  [3, 5, 8],
    5:  [0, 4, 6, 9],
    6:  [1, 5, 7, 10],
    7:  [2, 6, 11],
    8:  [4, 9, 12],
    9:  [5, 8, 10, 13],
    10: [6, 9, 11, 14],
    11: [7, 10, 15],
    12: [8, 13],
    13: [9, 12, 14],
    14: [10, 13, 15, 17],
    15: [11, 14, 16, 18],
    16: [15, 19],
    17: [ 18,23],
    18: [17, 19, 24],
    19: [18, 20, 25],
    20: [ 19, 21, 26],
    21: [20, 27],
    22: [ 23, 28],
    23: [ 22, 24, 29],
    24: [23, 25, 30],
    25: [19, 24, 26, 31],
    26: [20, 25, 27, 32],
    27: [21, 26, 33],
    28: [22, 29],
    29: [23, 28, 30, 34],
    30: [24, 29, 31, 35],
    31: [25, 30, 32, 36],
    32: [26, 31, 33, 37],
    33: [27, 32],
    34: [29, 35, 38],
    35: [30, 34, 36, 39],
    36: [31, 35, 37, 40],
    37: [32, 36, 41],
    38: [34, 39, 43],
    39: [35, 38, 40, 44],
    40: [36, 39, 41, 45],
    41: [37, 40, 42, 46],
    42: [41, 47],
    43: [38, 44],
    44: [39, 43, 45,48],
    45: [40, 44, 46, 49],
    46: [41, 45, 47, 50],
    47: [42, 46,51],
    48: [ 49],
    49: [ 48, 50],
    50: [ 49, 51],
    51: [ 50],
}

connection = []
edge_attribute = []

# build directed edges from adjacency
for i, neighbors in adjacency.items():
    for j in neighbors:
        connection.append((i, j))
        edge_attribute.append(distance_Base_to_Base_df.loc[j, f"base{i}"])

edge_index_base = th.tensor(connection, dtype=th.long).t().contiguous()
edge_attribute_base = np.array(edge_attribute).reshape(-1, 1)
edge_attribute_base_normalized = th.tensor(
    edge_attribute_base / np.max(edge_attribute_base),
    dtype=th.float
)

# make undirected (handles duplicates automatically)
edge_index_base, edge_attribute_base_normalized = to_undirected(
    edge_index_base,
    edge_attribute_base_normalized,
    num_nodes=NUM_AMBO_BASES
)

                                                        
class CustomGAT(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, 
                 features_dim: int = 52*64, 
                 edge_index_base : th.tensor = edge_index_base,
                 edge_attribute_base_normalized: th.tensor = edge_attribute_base_normalized
                 ):
        
        super().__init__(observation_space, features_dim)

        self.register_buffer('edge_index_base', edge_index_base) # dyamic memory allocation (GPU, CPU)
        self.register_buffer('edge_attribute_base_normalized', edge_attribute_base_normalized)

        self.GAT = GATConv(in_channels = 6,
                    out_channels = 64, 
                    heads = 1,
                    concat=False,  
                    add_self_loops = True, 
                    fill_value = 0,
                    edge_dim = 1, 
                    bias = True)
                    

    def edge_index_batching(self,edge_index : th.Tensor,batch_size ):
        edges =[]
        for i in range(batch_size):
            edge = edge_index+ i * NUM_AMBO_BASES
            edges.append(edge)   
        return th.cat(edges, dim=1) 


    def forward(self, observations: th.Tensor) -> th.Tensor:

        batch_size = observations.shape[0]
       
        x_base = observations.reshape(batch_size*NUM_AMBO_BASES,6)
        
        
        # batching edge index
        batched_edge_index_base = self.edge_index_batching(self.edge_index_base,batch_size)
        
        #batching edge attr
        batched_edge_attr_base = self.edge_attribute_base_normalized.repeat(batch_size, 1)

        # base node embedding
        x_base_embed = self.GAT(x_base, batched_edge_index_base, batched_edge_attr_base)
        x_base_embed = F.relu(x_base_embed)
        x_base_embed = x_base_embed.reshape(batch_size,-1)

        return x_base_embed

#########################################################################################################

# import torch as th
# from gymnasium import spaces
# from torch_geometric.data import Data
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# from torch_geometric.nn import GATConv
# from torch_geometric.utils import to_undirected
# import numpy as np
# import pandas as pd
# import torch.nn.functional as F

# distance_Base_to_Base_df = pd.read_csv("data/distance_base_to_base.csv")
# NUM_AMBO_BASES = distance_Base_to_Base_df.shape[0]


# # base 
# # Build edge_index and edge_attribute from manual adjacency
# adjacency = {
#     0:  [1, 5],
#     1:  [0, 2, 6],
#     2:  [1, 7],
#     3:  [4],
#     4:  [3, 5, 8],
#     5:  [0, 4, 6, 9],
#     6:  [1, 5, 7, 10],
#     7:  [2, 6, 11],
#     8:  [4, 9, 12],
#     9:  [5, 8, 10, 13],
#     10: [6, 9, 11, 14],
#     11: [7, 10, 15],
#     12: [8, 13],
#     13: [9, 12, 14],
#     14: [10, 13, 15, 17],
#     15: [11, 14, 16, 18],
#     16: [15, 19],
#     17: [ 18,23],
#     18: [17, 19, 24],
#     19: [18, 20, 25],
#     20: [ 19, 21, 26],
#     21: [20, 27],
#     22: [ 23, 28],
#     23: [ 22, 24, 29],
#     24: [23, 25, 30],
#     25: [19, 24, 26, 31],
#     26: [20, 25, 27, 32],
#     27: [21, 26, 33],
#     28: [22, 29],
#     29: [23, 28, 30, 34],
#     30: [24, 29, 31, 35],
#     31: [25, 30, 32, 36],
#     32: [26, 31, 33, 37],
#     33: [27, 32],
#     34: [29, 35, 38],
#     35: [30, 34, 36, 39],
#     36: [31, 35, 37, 40],
#     37: [32, 36, 41],
#     38: [34, 39, 43],
#     39: [35, 38, 40, 44],
#     40: [36, 39, 41, 45],
#     41: [37, 40, 42, 46],
#     42: [41, 47],
#     43: [38, 44],
#     44: [39, 43, 45,48],
#     45: [40, 44, 46, 49],
#     46: [41, 45, 47, 50],
#     47: [42, 46,51],
#     48: [ 49],
#     49: [ 48, 50],
#     50: [ 49, 51],
#     51: [ 50],
# }

# connection = []
# edge_attribute = []

# # build directed edges from adjacency
# for i, neighbors in adjacency.items():
#     for j in neighbors:
#         connection.append((i, j))
#         edge_attribute.append(distance_Base_to_Base_df.loc[j, f"base{i}"])

# edge_index_base = th.tensor(connection, dtype=th.long).t().contiguous()
# edge_attribute_base = np.array(edge_attribute).reshape(-1, 1)
# edge_attribute_base_normalized = th.tensor(
#     edge_attribute_base / np.max(edge_attribute_base),
#     dtype=th.float
# )


# # decision node
# edge_index_decision = th.stack([                #self loop at last digit
#     th.arange(NUM_AMBO_BASES, dtype=th.long) ,
#     th.full((NUM_AMBO_BASES,), NUM_AMBO_BASES, dtype=th.long)
                        
# ], dim=0)                                                         

# edge_index_full = th.cat([edge_index_base,edge_index_decision],dim = 1 )


# class CustomGAT(BaseFeaturesExtractor):


#     def __init__(self, observation_space: spaces.Box, 
#                  features_dim: int = 53*64, 
#                  edge_index_full : th.tensor = edge_index_full,
#                  edge_attribute_base_normalized: th.tensor = edge_attribute_base_normalized,
#                  ):
        
#         super().__init__(observation_space, features_dim)

#         self.register_buffer('edge_index_full', edge_index_full) # dyamic memory allocation (GPU, CPU)
#         self.register_buffer('edge_attribute_base_normalized', edge_attribute_base_normalized)


#         self.GAT = GATConv(in_channels = 5,
#                     out_channels = 64, 
#                     heads = 1,
#                     concat=False,  
#                     add_self_loops = True,
#                     fill_value = 0, 
#                     edge_dim = 1, 
#                     bias = True)
        
                    

#     def edge_batching(self,edge_index : th.Tensor,batch_size : int, offset : int):
#         edges =[]
#         for i in range(batch_size):
#             edge = edge_index+ i * offset
#             edges.append(edge)   
#         return th.cat(edges, dim=1) 


#     def forward(self, observations: th.Tensor) -> th.Tensor:

#         batch_size = observations.shape[0]
#         num_nodes = NUM_AMBO_BASES+1
        
#         # feature 
#         x_base = observations[:,:,1:6]
#         x_decision = x_base.mean(dim=1, keepdim=True)
#         x_full = th.cat([x_base,x_decision],dim=1)
#         x_dim = x_full.shape[1]
#         x_full = x_full.reshape(batch_size*x_dim,5)

#         # edge index
#         edge_full = self.edge_index_full
#         batched_edge_full = self.edge_batching(edge_full, batch_size ,num_nodes)

#         # edge attr
#         attr_base = self.edge_attribute_base_normalized.reshape(1,-1,1).expand(batch_size,-1,-1)
#         attr_decision = observations[:,:,0].reshape(batch_size,NUM_AMBO_BASES,1)
#         attr_full = th.cat([attr_base,attr_decision],dim = 1)
#         dim = attr_full.shape[1]
#         attr_full = attr_full.reshape(batch_size*dim,1)
        
#         # bidirectional graph
#         batched_edge_full,attr_full = to_undirected(batched_edge_full,
#                                                     attr_full,
#                                                     num_nodes=num_nodes*batch_size
#                                                     )

#         # node embedding
#         x_embed = self.GAT(x_full, batched_edge_full, attr_full)
#         x_embed = F.relu(x_embed)
#         x_embed = x_embed.reshape(batch_size,-1)
        
#         return x_embed


# ############################################################################################################################
# import torch as th
# from gymnasium import spaces
# from torch_geometric.data import Data
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# from torch_geometric.nn import GATConv
# from torch_geometric.utils import to_undirected
# import numpy as np
# import pandas as pd
# import torch.nn.functional as F

# distance_Base_to_Base_df = pd.read_csv("data/distance_base_to_base.csv")
# NUM_AMBO_BASES = distance_Base_to_Base_df.shape[0]


# # base 
# # Build edge_index and edge_attribute from manual adjacency
# adjacency = {
#     0:  [1, 5],
#     1:  [0, 2, 6],
#     2:  [1, 7],
#     3:  [4],
#     4:  [3, 5, 8],
#     5:  [0, 4, 6, 9],
#     6:  [1, 5, 7, 10],
#     7:  [2, 6, 11],
#     8:  [4, 9, 12],
#     9:  [5, 8, 10, 13],
#     10: [6, 9, 11, 14],
#     11: [7, 10, 15],
#     12: [8, 13],
#     13: [9, 12, 14],
#     14: [10, 13, 15, 17],
#     15: [11, 14, 16, 18],
#     16: [15, 19],
#     17: [ 18,23],
#     18: [17, 19, 24],
#     19: [18, 20, 25],
#     20: [ 19, 21, 26],
#     21: [20, 27],
#     22: [ 23, 28],
#     23: [ 22, 24, 29],
#     24: [23, 25, 30],
#     25: [19, 24, 26, 31],
#     26: [20, 25, 27, 32],
#     27: [21, 26, 33],
#     28: [22, 29],
#     29: [23, 28, 30, 34],
#     30: [24, 29, 31, 35],
#     31: [25, 30, 32, 36],
#     32: [26, 31, 33, 37],
#     33: [27, 32],
#     34: [29, 35, 38],
#     35: [30, 34, 36, 39],
#     36: [31, 35, 37, 40],
#     37: [32, 36, 41],
#     38: [34, 39, 43],
#     39: [35, 38, 40, 44],
#     40: [36, 39, 41, 45],
#     41: [37, 40, 42, 46],
#     42: [41, 47],
#     43: [38, 44],
#     44: [39, 43, 45,48],
#     45: [40, 44, 46, 49],
#     46: [41, 45, 47, 50],
#     47: [42, 46,51],
#     48: [ 49],
#     49: [ 48, 50],
#     50: [ 49, 51],
#     51: [ 50],
# }

# connection = []
# edge_attribute = []

# # build directed edges from adjacency
# for i, neighbors in adjacency.items():
#     for j in neighbors:
#         connection.append((i, j))
#         edge_attribute.append(distance_Base_to_Base_df.loc[j, f"base{i}"])

# edge_index_base = th.tensor(connection, dtype=th.long).t().contiguous()
# edge_attribute_base = np.array(edge_attribute).reshape(-1, 1)
# edge_attribute_base_normalized = th.tensor(
#     edge_attribute_base / np.max(edge_attribute_base),
#     dtype=th.float
# )

# # make undirected (handles duplicates automatically)
# edge_index_base, edge_attribute_base_normalized = to_undirected(
#     edge_index_base,
#     edge_attribute_base_normalized,
#     num_nodes=NUM_AMBO_BASES
# )


# # decision node
# edge_index_decision = th.stack([
#     th.arange(NUM_AMBO_BASES, dtype=th.long) ,
#     th.full((NUM_AMBO_BASES,), NUM_AMBO_BASES, dtype=th.long)
                        
# ], dim=0) 
                                                        
# class CustomGAT(BaseFeaturesExtractor):
#     """
#     :param observation_space: (gym.Space)
#     :param features_dim: (int) Number of features extracted.
#         This corresponds to the number of unit for the last layer.
#     """

#     def __init__(self, observation_space: spaces.Box, 
#                  features_dim: int =1024+52*64, 
#                  edge_index_base : th.tensor = edge_index_base,
#                  edge_attribute_base_normalized: th.tensor = edge_attribute_base_normalized,
#                  edge_index_decision: th.tensor = edge_index_decision):
        
#         super().__init__(observation_space, features_dim)

#         self.register_buffer('edge_index_base', edge_index_base) # dyamic memory allocation (GPU, CPU)
#         self.register_buffer('edge_attribute_base_normalized', edge_attribute_base_normalized)
#         self.register_buffer('edge_index_decision', edge_index_decision)

#         self.GAT_base = GATConv(in_channels = 6,
#                     out_channels = 64, 
#                     heads = 1,
#                     concat=False,  
#                     add_self_loops = True, 
#                     fill_value = 0,
#                     edge_dim = 1, 
#                     bias = True)
                    
#         self.GAT_decision = GATConv(in_channels = 64,
#                     out_channels = 1024, 
#                     heads = 1,
#                     concat=False,  
#                     add_self_loops = True,
#                     fill_value = 0,
#                     edge_dim = 1, 
#                     bias = True)

#     def edge_index_batching(self,edge_index : th.Tensor,batch_size, offset :int ):
#         edges =[]
#         for i in range(batch_size):
#             edge = edge_index+ i * offset
#             edges.append(edge)   
#         return th.cat(edges, dim=1) 


#     def forward(self, observations: th.Tensor) -> th.Tensor:

#         batch_size = observations.shape[0]
       
#         x_base = observations.reshape(batch_size*NUM_AMBO_BASES,6)
        
        
#         # batching edge index
#         batched_edge_index_base = self.edge_index_batching(self.edge_index_base,batch_size,NUM_AMBO_BASES)
#         batched_edge_index_decision = self.edge_index_batching(self.edge_index_decision,batch_size,NUM_AMBO_BASES+1)

#         #batching edge attr
#         batched_edge_attr_base = self.edge_attribute_base_normalized.repeat(batch_size, 1)
#         batched_edge_attr_decision = observations[:,:,0].reshape(batch_size*NUM_AMBO_BASES,1)

#         # base node embedding
#         x_base_embed = self.GAT_base(x_base, batched_edge_index_base, batched_edge_attr_base)
#         x_base_embed = F.relu(x_base_embed)

#         # adding decision node dummy
#         x_base_embed = x_base_embed.reshape(batch_size, NUM_AMBO_BASES, 64)  # (batch, 52, 64)
#         # dummy = th.zeros(batch_size, 1, 64, device=observations.device)       # (batch, 1, 64)
#         dummy = x_base_embed.mean(dim=1,keepdim=True)
#         x_full = th.cat([x_base_embed, dummy], dim=1)                         # (batch, 53, 64)
#         x_full = x_full.reshape(batch_size * (NUM_AMBO_BASES + 1), 64)        # (batch*53, 64)

#         # Decison node embedding
#         x_decision_embed = self.GAT_decision(x_full,batched_edge_index_decision,batched_edge_attr_decision)
#         x_decision_embed = F.relu(x_decision_embed)

#         # extract only decision node embedding (last node of each batch)
#         x_decision_embed = x_decision_embed.reshape(batch_size, NUM_AMBO_BASES + 1,1024)
#         decision_only = x_decision_embed[:, -1, :]  

#         state_embed = th.cat([x_base_embed.reshape(batch_size,-1),decision_only.reshape(batch_size,-1)], dim = 1)


#         return state_embed
