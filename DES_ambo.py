import simpy
import numpy as np
from collections import defaultdict

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register


class DES_ambo(gym.Env):
    metadata = {
                'render_modes': [],  #  no render modes supported
                'render_fps': 0      
                }
    def __init__(self,
                accident_rate = None,
                accident_rate_pred = None,
                distance_Base_to_Incident_df = None, 
                distance_Hospital_to_Base_df = None,
                nearest_place = None,
                init_ambulances_per_base_dict = None,
                run_until = 1440 ,
                trace=False,
                test = False,
                flat_feature = False):


        # Inherit from super class
        super(DES_ambo, self).__init__()

        self.trace_enabled = trace
        self.MINS_PER_DAY = 1440 # minute
        self.AMBULANCE_SPEED = 1000 # meter/min 

        #  Data
        self.accident_rate = accident_rate
        self.distance_Base_to_Incident_df = distance_Base_to_Incident_df
        self.distance_Hospital_to_Base_df = distance_Hospital_to_Base_df
        self.nearest_place = nearest_place

        # Initialize ambulances 
        self.init_ambulances_per_base_dict = init_ambulances_per_base_dict
        self.NUM_AMBULANCE_BASES = len(self.init_ambulances_per_base_dict)
        self.MAX_CAPACITY = 20 # max num of ambulances at each base
        self.TOTAL_AMBO = sum(self.init_ambulances_per_base_dict.values()) 
        self.max_incident_num = 4.2 # for dynamic scaling 

        # For testing return to own base behavior
        self.test = test

        # flat or graph features
        self.flat_feature = flat_feature

        # How many minutes environment run
        self.run_until = run_until
        self.max_incidents = 0 # arbitray large no


        # # un command to find mean demand at each period
        # self.now_incident_num = []
        # for _ in range(12):
        #     self.now_incident_num.append(np.zeros(self.NUM_AMBULANCE_BASES) )

        # Load accident_rate_pred
        self.accident_rate_pred = accident_rate_pred

        # Set action space
        if self.test:num_action = 5
        else: num_action = self.NUM_AMBULANCE_BASES
        self.action_space = spaces.Discrete(num_action) 

        if not self.flat_feature: # for GAT
            self.observation_space = spaces.Box(
                                                low=0, high=1,
                                                shape=(self.NUM_AMBULANCE_BASES, 6),  
                                                dtype=np.float32
                                        )

        else:
            self.observation_space = spaces.Box(
                                                low=0, high=1,
                                                shape=(self.NUM_AMBULANCE_BASES*6,),  
                                                dtype=np.float32
                                        )
                                            

    def reset(self,seed = None , options = None ):
        #random number generator
        if not seed:
            seed = np.random.randint(0, 1000000)

        else:
            seed = seed 
        
        self.rng = np.random.default_rng(seed)

        # Statistics collection
        self.time_from_call_to_incident_arr = []
        self.relocation_travel_time_arr = []
        self.time_from_incident_to_hospital_dict = defaultdict(list)
        self.time_from_hospital_to_return_dict = defaultdict(list)
        self.time_from_call_to_hospital_dict = defaultdict(list)
        self.step_reward = {}
        self.start_time = {}
        self.reward = []

        #new simpy env
        self.env = simpy.Environment()
        self.incident_counter = 0
        self.step_count = 0

        self.period = 0

        # to track returning ambulances
        self.return_ambo = np.zeros(self.NUM_AMBULANCE_BASES)
        self.return_ambo_time = defaultdict(list)

        # for relocation travel time
        self.relocation_travel_times = np.zeros(self.NUM_AMBULANCE_BASES)

        #initialize events
        self.arrived_hospital = self.env.event()
        self.ambo_available = self.env.event()

        # store waiting incidents events
        self.waiting_incidents = []

        # pass into to step
        self.waiting_for_action_info  = None
        
        # Initialize Resources (ambulance bases)
        self.base_resources = {}
        for base_id, count in self.init_ambulances_per_base_dict.items():
            self.base_resources[base_id] = simpy.Store(
                self.env, 
                capacity=self.MAX_CAPACITY
            )
            initial_ambulances = [base_id for _ in range(int(count))]
            self.base_resources[base_id].items.extend(initial_ambulances)

        # generate incidents   
        self.incident_process = self.env.process(self.incident_generator())
        self.env.run(until=self.arrived_hospital)

        if self.waiting_for_action_info:
            observation = self._get_obs(self.waiting_for_action_info['dispatched_ambo_id'])  

        self.trace(f'first_obs : {observation}')
        info = self._get_info()
        
        return observation,info


    def trace(self, message):
        """Conditional tracing function"""
        if self.trace_enabled:
            print(f"{self.env.now:.2f}: {message}")

    
    def incident_generator(self):
        """Generate incidents at each location according to Poisson process"""
        for row in self.accident_rate.itertuples():
            if row.mean_rate !=0:
                yield self.env.timeout(0)
                incident_id = row.Index
                mean_interarrival = self.MINS_PER_DAY/ row.mean_rate 

                self.env.process(self._incident_process(incident_id, mean_interarrival))
    
    def _incident_process(self, incident_id, mean_interarrival):
        """Process for generating incidents at a specific location"""
        while True:
            # Exponential interarrival times
            arrival = mean_interarrival
            if self.env.now <= self.run_until:
                if ((self.env.now/60) >= 0 and (self.env.now/60) < 2): #1
                    self.period = 0
                    arrival = (mean_interarrival)* 2.5
                elif ((self.env.now/60) >=2 and (self.env.now/60) < 4): #2
                    self.period = 1
                    arrival = (mean_interarrival) * 3
                elif ((self.env.now/60) >=4 and (self.env.now/60) < 6): #3
                    self.period = 2
                    arrival = (mean_interarrival) * 2.5
                elif ((self.env.now/60) >=6 and (self.env.now/60) < 8): #4
                    self.period = 3
                    arrival = (mean_interarrival) * 1
                elif ((self.env.now/60) >=8 and (self.env.now/60) < 10): #5
                    self.period = 4
                    arrival = (mean_interarrival) * 0.1
                elif ((self.env.now/60) >=10 and (self.env.now/60) < 12): #6
                    self.period = 5
                    arrival = (mean_interarrival) *0.3
                elif ((self.env.now/60) >=12 and (self.env.now/60) < 14): #7
                    self.period = 6
                    arrival = (mean_interarrival) * 0.6
                elif ((self.env.now/60) >=14 and (self.env.now/60) < 16): #8
                    self.period = 7
                    arrival = (mean_interarrival) * 0.9
                elif ((self.env.now/60) >=16 and (self.env.now/60) < 18): #9
                    self.period = 8
                    arrival = (mean_interarrival) * 1
                elif ((self.env.now/60) >=18 and (self.env.now/60) < 20): #10
                    self.period = 9
                    arrival = (mean_interarrival) * 0.9
                elif ((self.env.now/60) >=20 and (self.env.now/60) < 22): #11
                    self.period = 10
                    arrival = (mean_interarrival) * 1.6
                else:
                    self.period = 11
                    arrival = (mean_interarrival) * 2 #12

            else:
                break

            inter_arrival = self.rng.exponential(arrival) 
            yield self.env.timeout(inter_arrival)
            
            # Count incident
            incident_no = self.incident_counter
            self.incident_counter += 1

            # stop updating max_inidents when one episode is complete.
            # This counting is to make sure simulation waits for all ambulance mission generated during simulation time to complete
            # and simulation stops generating incidents when simulation time is up but wait for ongoing missions to complete
            if self.env.now <= self.run_until:
                self.max_incidents = self.incident_counter
        
            #find free ambulance
            self.env.process(self.find_free_ambo(incident_no, incident_id))
            
    def find_free_ambo(self, incident_no, incident_id):
        """Process for finding avaliable base, if no bases are avalibale, waits for returned ambuances"""
        self.start_time[incident_no] = self.env.now

        # Find bases with available ambulances
        unavailable_bases = []
        available_bases = []
        for base_id, resource in self.base_resources.items():
            if len(resource.items) == 0:
                unavailable_bases.append(base_id)
            else:
                available_bases.append(base_id)

        # the case when there is avaliable base
        if available_bases:

            if not unavailable_bases: # all bases are avaliable
                nearest_base_id = self.nearest_place.loc[incident_id,'nearest_base']
                nearest_base_distance = self.distance_Base_to_Incident_df.loc[nearest_base_id,f'incident_{incident_id}']

            else: # some bases are avaliable
                distance_arr = np.array(self.distance_Base_to_Incident_df[f'incident_{incident_id}'])

                # assign big distances to unavaliable base
                for i in unavailable_bases:
                    distance_arr[i] = np.inf

                # find nearest ambulance
                nearest_base_id = np.argmin(np.array(distance_arr))
                nearest_base_distance = distance_arr[nearest_base_id]


            # Dispatch ambulance
            if nearest_base_distance == np.inf:
                print("Warning")
            self.trace(f"[Incident {incident_no}] : at location {incident_id} with Nearest available base: {nearest_base_id} (distance: {nearest_base_distance:.2f})")
            self.env.process(self.ambulance_dispatch(incident_no, 
                                                    incident_id,
                                                    nearest_base_id, 
                                                    nearest_base_distance,
                                                    ))


        # If there is no ambulance base available at the time of call, wait for the first available ambulance
        else:

            # Create an event, this event is triggered when there is an avaliable ambulance
            ambo_available = self.env.event()
            
            # Store the event, FIFO (fist waiting incident gets firt avaliable ambulance)
            self.waiting_incidents.append(ambo_available)
            
            self.trace(f"[Incident {incident_no}] at location {incident_id} is waiting for an ambulance")
            
            # Wait for the event to be triggered
            yield ambo_available
            
            # Find which bases now has an ambulance
            available_bases = []
            available_bases_distance = []
            for base_id, resource in self.base_resources.items():
                if len(resource.items) > 0:
                    available_bases.append(base_id)
                    available_bases_distance.append(self.distance_Base_to_Incident_df.loc[base_id,f"incident_{incident_id}"])
                    self.trace(f"[incident {incident_no}] Base {base_id} becomes available")

            # Find nearest base 
            nearest_base_index = np.argmin(np.array(available_bases_distance))
            nearest_base_distance = available_bases_distance[nearest_base_index]
            nearest_base_id = available_bases[nearest_base_index]
            self.trace(f"[Incident {incident_no}] at location {incident_id} with Nearest available base: {nearest_base_id} (distance: {nearest_base_distance:.2f})")
        
            # Dispatch ambulance
            self.env.process(self.ambulance_dispatch(incident_no, 
                                                    incident_id,
                                                    nearest_base_id, 
                                                    nearest_base_distance,
                                                    ))
            

    def ambulance_dispatch(self, incident_no, incident_id,base_id, distance):
        """Process for ambulance dispatch and transport to incident"""

        # Request ambulance from the base 
        base_resource = self.base_resources[base_id]
        dispatched_ambo_id = yield base_resource.get()

        # Travel to incident
        travel_time = distance / self.AMBULANCE_SPEED  
        yield self.env.timeout(travel_time)
        
        # Calculate time from call to incident
        time_from_call_to_incident = self.env.now - self.start_time.get(incident_no)
        
        # reward shaping
        reward_signal = - np.tanh(travel_time-8)
        self.reward.append(reward_signal)
        self.trace(f"reward-signal is {reward_signal}")

        #collect stats
        self.time_from_call_to_incident_arr.append(time_from_call_to_incident)
        self.trace(f"[Incident {incident_no}] :Ambulance arrived at incident:  form base: {base_id}(response time: {time_from_call_to_incident:.2f} min)")
        
        # query nearest hospital distance 
        distance = self.nearest_place.loc[incident_id,'nearest_hospital_distance']

        # Transport to hospital
        self.env.process(self.transport_to_hospital(incident_no,
                                                    incident_id,
                                                    base_id,
                                                    dispatched_ambo_id,
                                                    distance,
                                                    time_from_call_to_incident,
                                                    reward_signal
                                                    ))
        
    
    def transport_to_hospital(self, incident_no, incident_id,base_id,dispatched_ambo_id,distance,time_from_call_to_incident,reward_signal):
        """Process for transporting patient to hospital"""

        # Travel to hospital, some randomness are added to reduce the chance of the ambulances arrived to bases at the same exact time
        travel_time = (distance / (self.AMBULANCE_SPEED)) + (self.rng.random())
        yield self.env.timeout(travel_time+5) # 5 extra min is for service time of dispatching patient

        # time from hospital to bases (1 factor of MDP state) 
        self.relocation_travel_times = np.array(self.distance_Hospital_to_Base_df[f'incident_{incident_id}_hospital']/(self.AMBULANCE_SPEED*0.7))
        
        # Record time from incident to hospital
        time_from_incident_to_hospital = (self.env.now - self.start_time.get(incident_no)) - time_from_call_to_incident
        
        self.trace(f"[Incident {incident_no}] at incident id {incident_id} arrived at hospital with ambulance from base {base_id} Take time {time_from_call_to_incident+time_from_incident_to_hospital:.2f}")


        # Pass info to step function to request a decision about which base to relocate
        self.waiting_for_action_info = {
                                        "incident_no": incident_no,
                                        "incident_id": incident_id,
                                        "base_id": base_id,
                                        "dispatched_ambo_id": dispatched_ambo_id,
                                        "time_from_call_to_incident": time_from_call_to_incident,
                                        "time_from_incident_to_hospital": time_from_incident_to_hospital,
                                        "reward_signal": reward_signal
                                       }
        

        # goes to step and request action by triggering event
        if not self.arrived_hospital.triggered:
            self.arrived_hospital.succeed()

        
    #functions below are for comparing different policies, they are not used in training phase
    def return_to_own_base(self,incident_id,base_id):
        """ambulances returing to own base"""

        distance = self.distance_Hospital_to_Base_df.loc[base_id,f'incident_{incident_id}_hospital']
        
        return distance,base_id
    

    def return_to_initial_base(self,incident_id,dispatched_ambo_id):
        """ambulances returning to the base that ambulances are initially assigned to"""

        distance = self.distance_Hospital_to_Base_df.loc[dispatched_ambo_id,f'incident_{incident_id}_hospital']

        return distance,dispatched_ambo_id


    def back_to_nearest_base(self,incident_id):
        """ambulances returning to the nearest base to the hospital"""

        distance_from_hospital_to_Base = self.distance_Hospital_to_Base_df[f'incident_{incident_id}_hospital']
        nearest_base_id = np.argmin(np.array(distance_from_hospital_to_Base))
        distance = distance_from_hospital_to_Base[nearest_base_id]

        return distance,nearest_base_id
        

    def back_to_base_with_less_ambo(self,incident_id):
        """ambulances returing to bases with the least amount of ambulances"""

        num_ambulances = np.array([len(self.base_resources[base_id].items) for base_id in range(self.NUM_AMBULANCE_BASES)])
        min_value = np.min(num_ambulances)                     # find smallest number
        lowest_num_bases = np.where(num_ambulances == min_value)[0]  # get base ids
        chosen_base_id = np.random.choice(lowest_num_bases)
        distance = self.distance_Hospital_to_Base_df.loc[chosen_base_id,f'incident_{incident_id}_hospital']

        return distance,chosen_base_id


    def back_to_highest_expected_demand (self,incident_id):        
        """ambulances returing to bases with highest expected call"""

        expected_rate = np.array(self.accident_rate_pred[self.period])
        max_value = np.max(expected_rate)
        highest_rate_incident_area = np.where(expected_rate == max_value)[0]  # extract array
        chosen_incident_area = np.random.choice(highest_rate_incident_area)
        base_id = self.nearest_place.loc[chosen_incident_area,'nearest_base']
        distance = self.distance_Hospital_to_Base_df.loc[base_id,f'incident_{incident_id}_hospital'] 
            
        return distance,base_id

    def relocate_base(self, incident_no, incident_id,base_id,dispatched_ambo_id,action,time_from_call_to_incident, time_from_incident_to_hospital,reward_signal):
        """Process for returning ambulance to base according to chosen action based of state"""

        if not self.test:
            relocate_base_id = action
            distance = self.distance_Hospital_to_Base_df.loc[relocate_base_id,f'incident_{incident_id}_hospital']

        else:
            # this is only for testing purposes, not used in training, actions are not base_id, just dummy values
            if action == 0:
                distance,relocate_base_id = self.return_to_own_base(incident_id,base_id)

            elif action == 1:
                distance,relocate_base_id = self.back_to_nearest_base(incident_id)

            elif action == 2:
                distance,relocate_base_id = self.back_to_base_with_less_ambo(incident_id)
                
            elif action == 3:
                distance,relocate_base_id = self.back_to_highest_expected_demand(incident_id)

            else:
                distance,relocate_base_id = self.return_to_initial_base(incident_id,dispatched_ambo_id)

        # Travel back to base
        travel_time = (distance / (self.AMBULANCE_SPEED*0.7)) + (self.rng.random())

        # Collect expected relocation finish time
        self.return_ambo_time[relocate_base_id].append(self.env.now+travel_time+5)
        yield self.env.timeout(travel_time+5) #5 extra min is for crew prepareation time before ambulances become avaliable again
        
        self.trace(f"[Incident no {incident_no}] Ambulance return to Base {relocate_base_id}:" )
        self.trace(f"Total time spent {self.env.now - self.start_time.get(incident_no)}")

        del self.start_time[incident_no] # delete value--no longer in use

        # Delete relocation time record for already arrived ambulances
        self.return_ambo_time[relocate_base_id] = [x for x in self.return_ambo_time[relocate_base_id] if x > self.env.now ]
                
        # Return ambulance  resource to base
        target_base_resource = self.base_resources[relocate_base_id]
        yield target_base_resource.put(dispatched_ambo_id)

        self.step_count = self.step_count+1

        # record relocation time
        self.relocation_travel_time_arr.append(travel_time)


        # triggered this only if there is waiting incidents
        # if there is waiting accidents, returned ambulance is assigned to the incident in FIFO
        if self.waiting_incidents:
            event_to_trig = self.waiting_incidents.pop(0)
            if not event_to_trig.triggered:
                event_to_trig.succeed()


    def _get_info(self):
        """return statistics"""
        return {"pick_up_times": self.time_from_call_to_incident_arr,"relocation_times": self.relocation_travel_time_arr,"total_incidents": self.max_incidents}

        
    def _get_obs(self,own_base_id): 
        """return observation"""
        
        #factor 1 (number of ambulances)
        num_ambulances_at_each_base = [len(self.base_resources[base_id].items) for base_id in range(self.NUM_AMBULANCE_BASES)]
        ambo_count = np.clip((np.array(num_ambulances_at_each_base, dtype=np.float32) / 5) ,0,1)

        #factor 2 (demand forecast)
        demand_forecast = np.clip((np.array(self.accident_rate_pred[self.period], dtype=np.float32)/ self.max_incident_num),0,1)

        # factor 3 (expected relocation finish time)
        # A large constant time to use for padding (30 min is fiest filled for every slot)
        padding_time = 30.0 
        
        returning_times_state = np.full(
            (self.NUM_AMBULANCE_BASES, 3), 
            padding_time
        )

        for base_id in range(self.NUM_AMBULANCE_BASES):
            
            arrival_time_list = self.return_ambo_time.get(base_id, []) # returns empth list if there is no base id in the key
            times_remaining = [
                t - self.env.now for t in arrival_time_list if (t - self.env.now) > 0
            ]
            
            if times_remaining:
                times_remaining.sort()
                num_to_take = min(len(times_remaining), 3) # take max 3 values
                nearest_times = times_remaining[:num_to_take]
                for i in range(num_to_take):
                    returning_times_state[base_id, i] = nearest_times[i] # add values

        # scale
        scaled_returning_times_state = np.clip(
                                                returning_times_state / padding_time, 
                                                0, 
                                                1
                                                )
        
        # factor 4 (expected relocation complete time)
        if np.max(self.relocation_travel_times) != 0:
            relocation_travel_times = np.clip((np.array(self.relocation_travel_times, dtype=np.float32)/ np.max(self.relocation_travel_times)),0,1)
        else:
            relocation_travel_times = np.clip(np.array(self.relocation_travel_times, dtype=np.float32),0,1)

        if not self.flat_feature:
            observation = np.concatenate([ relocation_travel_times.reshape(-1, 1),  # (52, 1)
                                            ambo_count.reshape(-1, 1),              # (52, 1)
                                            demand_forecast.reshape(-1,1),          # (52, 1)
                                            scaled_returning_times_state            # (52, 3)
                                        ], axis=1).astype(np.float32)               # (52, 6)
        else:
            observation = np.concatenate([  ambo_count.reshape(-1, 1),
                                            demand_forecast.reshape(-1, 1),
                                            scaled_returning_times_state,
                                            relocation_travel_times.reshape(-1, 1)
                                        ], axis=1).flatten().astype(np.float32)  # flatten 
        return observation
    
    def step(self, action):

        # this function is called when self.arrived_hospital is triggered
        self.reward = []
        self.trace(f"Action received: {action}")
        
        # create new event to retrigger again
        self.arrived_hospital = self.env.event()

        if self.waiting_for_action_info:
            # Unpack the info for the waiting ambulance
            info_1 = self.waiting_for_action_info
            
            # Start the relocation process in the background
            self.env.process(self.relocate_base(
                info_1["incident_no"], 
                info_1["incident_id"],
                info_1["base_id"], 
                info_1["dispatched_ambo_id"],
                action, # Pass the chosen base
                info_1["time_from_call_to_incident"],
                info_1["time_from_incident_to_hospital"],
                info_1["reward_signal"]
            ))
            self.waiting_for_action_info = None # Clear the waiting info

        # run 
        self.env.run(until=self.arrived_hospital)
        
        # Pass reward
        if not self.reward: # if no ambulances arrived at incident area during each step 
            reward = 0

        else: 
            reward = np.sum(np.array(self.reward)) # there can be multiple ambulances already arrived at incident area during each step
        self.trace(f"In step reward is {reward}")

        # Get observation
        if self.waiting_for_action_info: # new_info is collected
            observation = self._get_obs(self.waiting_for_action_info['dispatched_ambo_id'])  
        self.trace(f"Now state is {observation}")

        # terminate when simulation time is conplete and there is no pending ambulances still no mission
        terminated = (self.step_count >= self.max_incidents ) and (self.env.now >= self.run_until)

        truncated = False #never fail - step failes only if simulation time is completed, no truncation

        # Get information of each day
        if terminated: 
            info = self._get_info() 

        else:
            info = {} #no info is passed if simulation is not finished yet

        self.trace(f"total incident count : {self.max_incidents}")
        self.trace(f"total step count : {self.step_count}")
        
        return observation, reward, terminated, truncated, info
        
    
register(
    id="DES_ambo/DES_ambo_map-train",
    entry_point=DES_ambo,  
    kwargs={
        'accident_rate' : None,
        'accident_rate_pred' : None,
        'distance_Base_to_Incident_df' : None,
        'distance_Hospital_to_Base_df': None,
        'nearest_place': None,
        'init_ambulances_per_base_dict': None,
        'run_until' :1440,
        'trace' : False,
        'test' : False,
        'flat_feature' : False

    }
)
