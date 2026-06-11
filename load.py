# # Test number of ambulance
# import gymnasium as gym
# import torch
# import DES_ambo
# from stable_baselines3 import PPO
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import pickle
# from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# from stable_baselines3.common.monitor import Monitor

# # --- CONFIGURATION ---
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Training on: {device}")

# # Define directories
# BASE_DIR = "data"
# RESULTS_DIR = "results"  # Folder to save images and csvs
# os.makedirs(RESULTS_DIR, exist_ok=True)

# file_paths = {
#         "accident_rate": "data/accident_rate.csv",
#         "distance_Base_to_Incident_df": "data/distance_base_to_incident.csv",
#         "distance_Hospital_to_Base_df": "data/distance_hospital_to_base.csv",
#         "nearest_place" : "data/nearest_places_data.csv",
#         "ambulance_initialization" : "data/ambulance_initialization.csv"
#     }
# # --- LOAD DATA ---
# # Check if files exist
# for name, path in file_paths.items():
#     if not os.path.exists(path):
#         print(f"Error: {name} file not found at {path}")

# accident_rate = pd.read_csv(file_paths["accident_rate"])
# distance_Base_to_Incident_df = pd.read_csv(file_paths["distance_Base_to_Incident_df"])
# distance_Hospital_to_Base_df = pd.read_csv(file_paths["distance_Hospital_to_Base_df"])
# nearest_place = pd.read_csv(file_paths["nearest_place"])
# ambulance_initialization = pd.read_csv(file_paths["ambulance_initialization"])
# ambulance_initialization_dict = ambulance_initialization.to_dict()['initial_ambulances']

# # Load prediction pickle
# with open('data/incident_pred.pkl', 'rb') as f:
#     accident_rate_pred = pickle.load(f)

# # --- HELPER FUNCTIONS ---

# def get_stats_string(data_list):
#     """
#     Calculates Mean and 95% CI and returns string 'Mean +/- CI'
#     """
#     arr = np.array(data_list)
#     n = len(arr)
#     mean_val = np.mean(arr)
#     std_err = np.std(arr, ddof=1) / np.sqrt(n)
#     ci = 1.96 * std_err
#     return f"{mean_val:.2f} +/- {ci:.2f}"

# def create_plot_and_save(results_dict, title, filename, time=True):
#     """
#     Generates, shows, AND saves the plot.
#     """
#     labels = list(results_dict.keys())
#     data = [results_dict[label] for label in labels if results_dict.get(label) is not None and len(results_dict.get(label)) > 0]
#     labels_with_data = [label for label in labels if results_dict.get(label) is not None and len(results_dict.get(label)) > 0]
    
#     if not data:
#         return

#     # Calculate stats for labels
#     means = []
#     conf_intervals = []
#     for d in data:
#         arr = np.asanyarray(d)
#         n = len(arr)
#         mean_val = np.mean(arr)
#         std_err = np.std(arr, ddof=1) / np.sqrt(n)
#         ci = 1.96 * std_err
#         means.append(mean_val)
#         conf_intervals.append(ci)

#     plt.figure(figsize=(12, 8))
#     box = plt.boxplot(data, patch_artist=True, labels=labels_with_data, showmeans=True, meanline=True,
#                       medianprops={'linestyle':'-', 'color': 'black', 'linewidth': 2},
#                       meanprops={'linestyle':'--', 'color':'firebrick', 'linewidth': 2})
    
#     base_colors = ['lightgray'] * len(data)
#     if len(base_colors) > 0: base_colors[-1] = 'lightblue' # Highlight PPO
#     for patch, color in zip(box['boxes'], base_colors): patch.set_facecolor(color)
    
#     # Add text labels
#     x_ticks_positions = range(1, len(labels_with_data) + 1)
#     for x, mean_val, error in zip(x_ticks_positions, means, conf_intervals):
#         text_label = f'{mean_val:.2f}\n±{error:.2f}'
#         plt.text(x, mean_val + (mean_val * 0.05), text_label, ha='center', va='bottom', fontsize=10, fontweight='bold', color='firebrick') 

#     plt.xticks(rotation=45, ha="right") 
#     plt.title(title, fontsize=16)
#     plt.ylabel('Time (min)' if time else 'Ratio', fontsize=12)
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.legend([box["medians"][0], box["means"][0]], ['Median', 'Mean (95% CI)'], loc='upper right')
    
#     if time:
#         max_val = max([np.max(d) for d in data])
#         plt.ylim(0, max_val + (max_val*0.15)) 
#     else:
#         plt.ylim(0, 1.1)
    
#     plt.tight_layout()
#     # Save the plot
#     plt.savefig(os.path.join(RESULTS_DIR, filename))
#     plt.close() # Close to free memory
#     print(f"Saved plot: {filename}")

# def model_evaluate(test_episodes, env = None,model=None, base_line=None, random=False):
#     ratio_pick_up_store = []
#     pick_up_time_lst = []
#     relocation_time_lst = []
#     pick_up_time_95_threshold_lst = []
#     max_pick_up_time_lst = []

#     for i in range(test_episodes):
#         print(f'test rpisode : {i}')
#         obs = env.reset()
#         done = False
#         Reward = 0
        
#         while not done:
#             if model:
#                 if not random:
#                     action, _ = model.predict(obs, deterministic=True)
#                 else:
#                     # Logic for Random inside Vectorized Env
#                     action = [env.action_space.sample()]
#                 obs, reward, terminated, info = env.step(action)
#             else:
#                 # Logic for Heuristics
#                 action = base_line #dummy action
#                 obs, reward, terminated, _, info = env.step(int(action))

#             done = terminated
#             Reward += reward
            
#         # Extract Info
#         if model:
#             pick_up_time, relocation_time, total_incidents = info[0]["pick_up_times"], info[0]["relocation_times"], info[0]["total_incidents"]
#         else:
#             pick_up_time, relocation_time, total_incidents = info["pick_up_times"], info["relocation_times"], info["total_incidents"]

#         # Calculate metrics for this episode
#         on_time_count = np.sum(np.array(pick_up_time)<8)
#         ratio_pick_up_store.append(on_time_count/len(pick_up_time))
#         pick_up_time_lst.append(np.mean(np.array(pick_up_time)))
#         pick_up_time_95_threshold_lst.append(np.percentile(np.array(pick_up_time), 95))
#         max_pick_up_time_lst.append(np.max(np.array(pick_up_time)))
#         relocation_time_lst.append(np.mean(np.array(relocation_time)))

#     return (np.array(ratio_pick_up_store).flatten(),
#             np.array(relocation_time_lst),
#             np.array(pick_up_time_lst),
#             np.array(pick_up_time_95_threshold_lst).flatten(),
#             np.array(max_pick_up_time_lst))

# # --- MAIN EXECUTION LOOP ---

# def run_evaluation_suite():
    
#     # --- 1. DEFINE MODELS AND THEIR STATS ---
#     # FORMAT: (Model_Path, Stats_Path)
#     models_to_test = [

#         # (
#         #     "/home/thurein/ambo_allocate/integrate_map/25_finetuned_45M.zip", 
#         #     "/home/thurein/ambo_allocate/integrate_map/25_finetuned_45M_stat.pkl"
#         # ),
      
#         # (
#         #     "/home/thurein/ambo_allocate/integrate_map/30_finetuned_50M.zip", 
#         #     "/home/thurein/ambo_allocate/integrate_map/30_finetuned_50M_stat.pkl"
#         # ),
#         # (
#         #     "/home/thurein/ambo_allocate/integrate_map/PPO_finetuned_30M.zip", 
#         #     "/home/thurein/ambo_allocate/integrate_map/PPO_stats_finetuned_30M.pkl"
#         # ),
#         # (
#         #     "/home/thurein/ambo_allocate/integrate_map/40_finetuned_45M.zip", 
#         #     "/home/thurein/ambo_allocate/integrate_map/40_finetuned_45M_stat.pkl"
#         # ),
#         (
#             "flat.zip", 
#             "flat_stat.pkl"
#         ),
#     ]
    
#     TEST_EPI = 20
    
#     # Loop over tuples: (model, stats)
#     for i, (model_path, stats_path) in enumerate(models_to_test):
        
#         model_name = f"r2_Model_{i+1}_{os.path.basename(model_path)}" ################################
#         print(f"\n{'='*20}\nEvaluating: {model_name}\nUsing stats: {os.path.basename(stats_path)}\n{'='*20}")
        
#         # Check if model/stats files exist before running
#         if not os.path.exists(model_path) or not os.path.exists(stats_path):
#             print(f"Skipping {model_name}: File not found.")
#             continue

#         # --- A. SETUP ENVIRONMENTS ---
        
#         # 1. Trained PPO Environment (Uses SPECIFIC stats_path)
#         env_ppo = gym.make("DES_ambo/DES_ambo_map-train", accident_rate=accident_rate, accident_rate_pred=accident_rate_pred,
#                            distance_Base_to_Incident_df=distance_Base_to_Incident_df, distance_Hospital_to_Base_df=distance_Hospital_to_Base_df,
#                            nearest_place=nearest_place, init_ambulances_per_base_dict=ambulance_initialization_dict,
#                            run_until=1440, trace=False, test=False, flat_feature = True)
#         env_ppo = Monitor(env_ppo)
#         env_ppo = DummyVecEnv([lambda: env_ppo])
        
#         # LOAD SPECIFIC STATS
#         try:
#             norm_env = VecNormalize.load(stats_path, env_ppo)
#             norm_env.training = False
#             norm_env.norm_reward = False
#         except Exception as e:
#             print(f"Error loading stats file {stats_path}: {e}")
#             continue
        
#         # 2. Heuristic Environment (Test=True)
#         env_heuristic = gym.make("DES_ambo/DES_ambo_map-train", accident_rate=accident_rate, accident_rate_pred=accident_rate_pred,
#                                  distance_Base_to_Incident_df=distance_Base_to_Incident_df, distance_Hospital_to_Base_df=distance_Hospital_to_Base_df,
#                                  nearest_place=nearest_place, init_ambulances_per_base_dict=ambulance_initialization_dict,
#                                  run_until=1440, trace=False, test=True)
        
#         # 3. Random Baseline Environment (Test=False)
#         # env_random_base = gym.make("DES_ambo/DES_ambo_map-train", accident_rate=accident_rate, accident_rate_pred=accident_rate_pred,
#         #                            distance_Base_to_Incident_df=distance_Base_to_Incident_df, distance_Hospital_to_Base_df=distance_Hospital_to_Base_df,
#         #                            nearest_place=nearest_place, init_ambulances_per_base_dict=ambulance_initialization_dict,
#         #                            run_until=1440, trace=False, test=False)
#         # env_random_base = Monitor(env_random_base)
#         # env_random_base = DummyVecEnv([lambda: env_random_base])

#         # --- B. LOAD MODEL ---
#         try:
#             model = PPO.load(model_path, env=norm_env, device=device)
#         except Exception as e:
#             print(f"Failed to load model at {model_path}: {e}")
#             continue

#         # --- C. RUN EVALUATIONS ---
        
#         print("1. Running Trained PPO...")
#         ppo_res = model_evaluate(TEST_EPI, norm_env, model=model)
        
#         print("2. Running Random...")
#         rand_res = model_evaluate(TEST_EPI, norm_env, model=model, random=True)
        
#         print("3. Running Heuristics (NB)...")
#         nb_res = model_evaluate(TEST_EPI, env_heuristic, model=None, base_line=1)
        
#         print("4. Running Heuristics (RB)...")
#         rb_res = model_evaluate(TEST_EPI, env_heuristic, model=None, base_line=0)
        
#         print("5. Running Heuristics (LB)...")
#         lb_res = model_evaluate(TEST_EPI, env_heuristic, model=None, base_line=2)

#         # Unpack results for plotting (Indices: 0=Ratio, 1=Reloc, 2=Pickup, 3=95%, 4=Max)
#         results_map = {
#             "Ratio": [rand_res[0], nb_res[0], rb_res[0], lb_res[0], ppo_res[0]],
#             "Relocation": [rand_res[1], nb_res[1], rb_res[1], lb_res[1], ppo_res[1]],
#             "PickUp": [rand_res[2], nb_res[2], rb_res[2], lb_res[2], ppo_res[2]],
#             "PickUp95": [rand_res[3], nb_res[3], rb_res[3], lb_res[3], ppo_res[3]],
#             "MaxPickUp": [rand_res[4], nb_res[4], rb_res[4], lb_res[4], ppo_res[4]],
#         }
#         # results_map = {
#         #     "Ratio": [ppo_res[0]],
#         #     "Relocation": [ppo_res[1]],
#         #     "PickUp": [ppo_res[2]],
#         #     "PickUp95": [ppo_res[3]],
#         #     "MaxPickUp": [ppo_res[4]],
#         # }
        
#         policy_names = ['Random','NB','DSM','LB','Trained_PPO']

#         # --- D. SAVE PLOTS ---
#         # Helper to build dict for plotting function
#         def build_dict(metric_key):
#             return {name: data for name, data in zip(policy_names, results_map[metric_key])}

#         create_plot_and_save(build_dict("Ratio"), f"Pick Up Ratio - {model_name}", f"ratio_{model_name}.png", time=False)
#         create_plot_and_save(build_dict("Relocation"), f"Relocation Time - {model_name}", f"reloc_{model_name}.png", time=True)
#         create_plot_and_save(build_dict("PickUp"), f"Avg Response Time - {model_name}", f"resp_avg_{model_name}.png", time=True)
#         create_plot_and_save(build_dict("PickUp95"), f"95% Response Time - {model_name}", f"resp_95_{model_name}.png", time=True)
#         create_plot_and_save(build_dict("MaxPickUp"), f"Max Response Time - {model_name}", f"resp_max_{model_name}.png", time=True)

#         # --- E. GENERATE SUMMARY CSV ---
#         print("Generating Summary Dataframe...")
        
#         # Structure for DataFrame
#         summary_data = {
#             "Metric": [
#                 "8 min threshold (Ratio)", 
#                 "Max. response time", 
#                 "Avg. response time", 
#                 "95 % threshold", 
#                 "relocation travel time"
#             ]
#         }
        
#         # Populate columns for each policy
#         for idx, policy in enumerate(policy_names):
#             col_data = []
#             col_data.append(get_stats_string(results_map["Ratio"][idx]))      # 8 min threshold
#             col_data.append(get_stats_string(results_map["MaxPickUp"][idx]))  # Max response
#             col_data.append(get_stats_string(results_map["PickUp"][idx]))     # Avg response
#             col_data.append(get_stats_string(results_map["PickUp95"][idx]))   # 95%
#             col_data.append(get_stats_string(results_map["Relocation"][idx])) # Relocation
            
#             summary_data[policy] = col_data

#         # Create DataFrame
#         df = pd.DataFrame(summary_data)
        
#         # Save CSV
#         csv_filename = os.path.join(RESULTS_DIR, f"summary_table_{model_name}.csv")
#         df.to_csv(csv_filename, index=False)
#         print(f"Saved summary CSV to: {csv_filename}")

# if __name__ == "__main__":
#     run_evaluation_suite()


import numpy as np
import pandas as pd
import gymnasium as gym
import torch
import DES_ambo
from stable_baselines3 import PPO
import pickle
import os
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── data loading ───────────────────────────────────────────────────
file_paths = {
    "accident_rate":             "data/accident_rate.csv",
    "distance_Base_to_Incident": "data/distance_base_to_incident.csv",
    "distance_Hospital_to_Base": "data/distance_hospital_to_base.csv",
    "nearest_place":             "data/nearest_places_data.csv",
    "ambulance_initialization":  "data/ambulance_initialization.csv"
}

accident_rate              = pd.read_csv(file_paths["accident_rate"])
distance_Base_to_Incident  = pd.read_csv(file_paths["distance_Base_to_Incident"])
distance_Hospital_to_Base  = pd.read_csv(file_paths["distance_Hospital_to_Base"])
nearest_place              = pd.read_csv(file_paths["nearest_place"])
ambulance_initialization   = pd.read_csv(file_paths["ambulance_initialization"])
ambulance_initialization_dict = ambulance_initialization.to_dict()['initial_ambulances']

with open('data/incident_pred.pkl', 'rb') as f:
    accident_rate_pred = pickle.load(f)

# ── model paths ────────────────────────────────────────────────────
FLAT_MODEL  = "flat.zip"
FLAT_STATS  = "flat_stat.pkl"
GAT_MODEL   = "self_loop.zip"
GAT_STATS   = "self_loop_stat.pkl"

TEST_EPI    = 500

# ── base env kwargs ────────────────────────────────────────────────
base_env_kwargs = dict(
    accident_rate                  = accident_rate,
    accident_rate_pred             = accident_rate_pred,
    distance_Base_to_Incident_df   = distance_Base_to_Incident,
    distance_Hospital_to_Base_df   = distance_Hospital_to_Base,
    nearest_place                  = nearest_place,
    init_ambulances_per_base_dict  = ambulance_initialization_dict,
    run_until                      = 1440,
    trace                          = False
)

# ── helper: make vectorized env ────────────────────────────────────
def make_vec_env(flat_feature=False):
    env = gym.make("DES_ambo/DES_ambo_map-train",
                   **base_env_kwargs, test=False,
                   flat_feature=flat_feature)
    env = Monitor(env)
    return DummyVecEnv([lambda: env])

def make_heuristic_env():
    return gym.make("DES_ambo/DES_ambo_map-train",
                    **base_env_kwargs, test=True)

# ── helper: load ppo model ─────────────────────────────────────────
def load_model(model_path, stats_path, flat_feature=False):
    vec_env  = make_vec_env(flat_feature=flat_feature)
    norm_env = VecNormalize.load(stats_path, vec_env)
    norm_env.training   = False
    norm_env.norm_reward = False
    model = PPO.load(model_path, env=norm_env, device=device)
    return model, norm_env

# ── evaluation function ────────────────────────────────────────────
def evaluate(test_episodes, env=None, model=None,
             heuristic_action=None, random=False):
    ratio_lst   = []
    pickup_lst  = []

    for ep in range(test_episodes):
        print(f"  episode {ep+1}/{test_episodes}")
        obs  = env.reset()
        done = False

        while not done:
            if model is not None:
                if random : action = [env.action_space.sample()]
                else: action, _ = model.predict(obs, deterministic= False)
                obs, reward, terminated, info = env.step(action)
                done = terminated
                if done:
                    pick_up_times = info[0]["pick_up_times"]
            else:
                obs, reward, terminated, _, info = env.step(int(heuristic_action))
                
                done = terminated
                if done:
                    pick_up_times = info["pick_up_times"]

        arr          = np.array(pick_up_times)
        on_time      = np.sum(arr < 8) / len(arr)
        avg_response = np.mean(arr)

        ratio_lst.append(on_time)
        pickup_lst.append(avg_response)

    return np.array(ratio_lst), np.array(pickup_lst)

# ── stats helper ───────────────────────────────────────────────────
def stats_str(arr):
    mean = np.mean(arr)
    ci   = 1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr))
    return f"{mean:.4f} ± {ci:.4f}"

# ── plot helper ────────────────────────────────────────────────────
def plot_metric(data_dict, title, ylabel, filename, ratio=False):
    labels = list(data_dict.keys())
    data   = list(data_dict.values())
    means  = [np.mean(d) for d in data]
    cis    = [1.96 * np.std(d, ddof=1) / np.sqrt(len(d)) for d in data]

    plt.figure(figsize=(10, 6))
    box = plt.boxplot(data, patch_artist=True, labels=labels,
                      showmeans=True, meanline=True,
                      medianprops={'color': 'black', 'linewidth': 2},
                      meanprops={'linestyle': '--', 'color': 'firebrick', 'linewidth': 2})

    colors = ['lightgray', 'lightgray', 'lightyellow', 'lightblue']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    if ratio: offset = 0.2
    else: offset = 4
    for x, mean, ci in zip(range(1, len(labels)+1), means, cis):
        plt.text(x, mean + offset, f'{mean:.2f}±{ci:.2f}',
                 ha='center', fontsize=9,
                 fontweight='bold', color='firebrick')

    plt.title(title, fontsize=14)
    plt.ylabel(ylabel, fontsize=12)
    plt.ylim(0, 1.1 if ratio else None)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()
    print(f"Saved: {filename}")

# ── main ───────────────────────────────────────────────────────────
if __name__ == "__main__":

    results = {}
    flat_model, flat_norm = load_model(FLAT_MODEL, FLAT_STATS, flat_feature=True)
    # 1. random
    print("\n[1/4] Evaluating Rnadom...")
    ratio, pickup = evaluate(TEST_EPI, env=flat_norm, model=flat_model,random=True)
    results["Random"] = (ratio, pickup)

    # 2. DSM 
    print("\n[2/4] Evaluating DSM...")
    env_h = make_heuristic_env()
    ratio, pickup = evaluate(TEST_EPI, env=env_h, heuristic_action=0)
    results["DSM"] = (ratio, pickup)

    # 1. Flat PPO
    print("\n[3/4] Evaluating Flat PPO...")
 
    ratio, pickup = evaluate(TEST_EPI, env=flat_norm, model=flat_model)
    results["Flat"] = (ratio, pickup)

    # 4. GAT PPO
    print("\n[4/4] Evaluating GAT PPO...")
    gat_model, gat_norm = load_model(GAT_MODEL, GAT_STATS, flat_feature=False)
    ratio, pickup = evaluate(TEST_EPI, env=gat_norm, model=gat_model)
    results["GAT PPO"] = (ratio, pickup)

    # ── plots ──────────────────────────────────────────────────────
    plot_metric(
        {k: v[0] for k, v in results.items()},
        "Pick-up Ratio Within 8 Minutes",
        "Ratio", "ratio_comparison.png", ratio=True
    )
    plot_metric(
        {k: v[1] for k, v in results.items()},
        "Average Response Time",
        "Time (min)", "response_time_comparison.png"
    )

    # ── summary table ──────────────────────────────────────────────
    summary = pd.DataFrame({
        "Policy": list(results.keys()),
        "Pick-up Ratio (Mean ± 95% CI)": [stats_str(v[0]) for v in results.values()],
        "Avg Response Time (Mean ± 95% CI)": [stats_str(v[1]) for v in results.values()]
    })

    csv_path = os.path.join(RESULTS_DIR, "summary_table.csv")
    summary.to_csv(csv_path, index=False)
    print(f"\nSummary saved to: {csv_path}")
    print(summary.to_string(index=False))