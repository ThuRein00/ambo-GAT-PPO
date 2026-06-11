import pulp
import numpy as np

class DSM:
    def __init__(self,mean_rate,nearest_place,distance_Base_to_Incident_df):

        self.mean_rate = mean_rate
        self.distance_Base_to_Incident_df = distance_Base_to_Incident_df
        self.nearest_place = nearest_place

    
    def find_covered_base (self,incident_id,r):
        covered_base = []
        for base_id in range(len(self.distance_Base_to_Incident_df)):
            distance = self.distance_Base_to_Incident_df.loc[base_id,f"incident_{incident_id}"]
            if distance <= r:
                covered_base.append(base_id)
        return covered_base

    def _get_neighborhood_set(self,r):
        neighborhood_set_dict = {}
        for incident_id in range(len(self.nearest_place)):
            neighborhood_set = self.find_covered_base(incident_id,r)
            neighborhood_set_dict[incident_id] = neighborhood_set
        return neighborhood_set_dict


    def solve(self,r1=8000,r2=8000*2,alpha = 0.5,total_ambulances=None,max_abmulances = {}):

        model = pulp.LpProblem("Double_Standard_Model", pulp.LpMaximize)
        
        d = self.mean_rate['mean_rate'].to_list()
        incident_id = [i for i in range(len(self.nearest_place))]
        print(incident_id)
        base_id = [i for i in range(len(self.distance_Base_to_Incident_df))]
        print(base_id)

        # Decision variables
        y = pulp.LpVariable.dicts("y", base_id, lowBound=0, cat="Integer")     # number of ambulances at site j
        x1 = pulp.LpVariable.dicts("x1", incident_id, lowBound=0, upBound=1, cat="Binary")  # covered once
        x2 = pulp.LpVariable.dicts("x2", incident_id, lowBound=0, upBound=1, cat="Binary")  # covered twice

        # neighborhood set
        Wi1 = self._get_neighborhood_set(r1)
        Wi2 = self._get_neighborhood_set(r2)

        # Objective: maximize demand covered twice within r1 and r2
        model += pulp.lpSum(d[i] * x2[i] for i in incident_id)

        # constraints
        for i in incident_id:   # 30
            neighborhood_set = Wi2[i]
            model += pulp.lpSum(y[j] for j in neighborhood_set) >= 1

        model += pulp.lpSum(d[i] * x1[i] for i in incident_id) >= pulp.lpSum(alpha * d[i] for i in incident_id) #31

        for i in incident_id: #32
            neighborhood_set = Wi1[i]
            model += pulp.lpSum(y[j] for j in neighborhood_set) >= x1[i] + x2[i]

        for i in incident_id: #33
            model += x2[i] <= x1[i]

        model += pulp.lpSum(y[j] for j in base_id) == total_ambulances #34

        for j in base_id:
            model += y[j] <= max_abmulances[j]
        
        model.solve(pulp.PULP_CBC_CMD(msg = False))
        
        ambulance_initialization = pd.DataFrame()
        num_ambulamces = []

        # RESULTS
        print("Status:", pulp.LpStatus[model.status])
        print("Objective (double-covered demand):", pulp.value(model.objective))

        print("\nAmbulance locations:")
        for j in base_id:
            if y[j].value() > 0:
                print(f"  Ambulance Base {j}: {y[j].value()} ambulances")
            num_ambulamces.append(y[j].value())
        ambulance_initialization['initial_ambulances'] = num_ambulamces
        ambulance_initialization.to_csv("data/ambulance_initialization.csv",index=False)

        print("\nDemand coverage:")
        for i in incident_id:
            print(f"  Demand {i}: x1={x1[i].value()}, x2={x2[i].value()}")


# Solve
import pandas as pd
import numpy as np
import os

file_paths = {
    "accident_rate": "data/accident_rate.csv",
    "distance_Base_to_Incident_df": "data/distance_base_to_incident.csv",
    "nearest_place" : "data/nearest_places_data.csv",
}

# Check if files exist
for name, path in file_paths.items():
    if not os.path.exists(path):
        print(f"Error: {name} file not found at {path}")
    else:
        print(f"Found {name} file at {path}")

# Load files 
accident_rate = pd.read_csv(file_paths["accident_rate"])
print(f"mean_rate = {np.sum(np.array(accident_rate['mean_rate']))}")
print(f"Sum of raw rates: {accident_rate['mean_rate'].sum()}")
print(f"Non-zero count: {(accident_rate['mean_rate'] != 0).sum()}")
print(f"After adjustment: {(accident_rate['mean_rate'] * (12/21.48)).sum()}")
accident_rate = accident_rate * (12/21.48) # avg rate across all periods
distance_Base_to_Incident_df = pd.read_csv(file_paths["distance_Base_to_Incident_df"])
nearest_place = pd.read_csv(file_paths["nearest_place"])
solver = DSM(accident_rate,nearest_place,distance_Base_to_Incident_df)

# Initialize simulation
max_ambulances = {}
for i in range(len(distance_Base_to_Incident_df)):
    max_ambulances[i] = 5   # max allowed ambulances at each base
solver.solve(total_ambulances=35,max_abmulances=max_ambulances)