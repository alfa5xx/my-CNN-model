import os
import sys
import traci
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from gymnasium import spaces
import random
import string
import sumolib



class SUMOEnv(gym.Env):
    def __init__(self):
        super(SUMOEnv, self).__init__()

        # ✅ Define action and observation space
        self.action_space = spaces.Discrete(5)  # Assuming 5 possible next edges
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)  # Example observation

        # ✅ Start SUMO before accessing edges
        sumo_cmd = ["sumo", "-c", "C:/Users/Administrator/Desktop/Prof.Mangini/Project2025/b.sumocfg", "--no-warnings"]
        traci.start(sumo_cmd)

        # ✅ Now retrieve edges AFTER SUMO is running
        self.edge_list = self.get_all_edges()

    def get_all_edges(self):
        """Retrieve all possible edges from SUMO network"""
        return [edge for edge in traci.edge.getIDList()]  # Fix: No need for `getID()`

    def step(self, action):
        """Apply action and return observation, reward, done, info"""
        if action < len(self.edge_list):  # Fix: Prevent index error
            traci.vehicle.changeTarget("veh0", self.edge_list[action])

        traci.simulationStep()

        # Example reward logic
        reward = -1  # Penalize for longer routes

        return self._get_observation(), reward, False, False, {}

    def reset(self, seed=None, options=None):
        """Reset environment"""
        traci.load(["-c", "C:/Users/Administrator/Desktop/Prof.Mangini/Project2025/b.sumocfg"])  # Fix: Reload SUMO on reset
        return self._get_observation(), {}

    def _get_observation(self):
        """Return example observation"""
        return np.random.rand(10)  # Replace with real SUMO features

# ✅ Create environment after SUMO is properly initialized
env = SUMOEnv()

# ✅ Train PPO agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# ✅ Save the model
model.save("ppo_sumo")

def generate_vehicle_id():
    """Generate a random 5-character vehicle ID."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=5))

def get_random_route(net):
    """Generate a random route based on SUMO network."""
    edges = list(net.getEdges())  # Get all edges in the network
    if len(edges) < 2:
        raise ValueError("Network must have at least two edges.")

    # ✅ Randomly select start and end edges (ensuring they are different)
    from_edge = random.choice(edges)
    to_edge = random.choice(edges)
    while to_edge == from_edge:  
        to_edge = random.choice(edges)

    # ✅ Find shortest path between selected edges
    path, _ = net.getShortestPath(from_edge, to_edge)

    # Convert to route ID list
    route_edges = [edge.getID() for edge in path]
    return route_edges if route_edges else [from_edge.getID(), to_edge.getID()]

def reset(self, seed=None, options=None):
    """Reset SUMO environment and generate a random route."""
    traci.load(["-c", "b.sumocfg"])  # ✅ Load SUMO config

    self.vehicle_id = generate_vehicle_id()  # ✅ Generate random vehicle ID

    # ✅ Load SUMO network file from the SUMO config
    net = sumolib.net.readNet("C:/Users/Administrator/Desktop/Prof.Mangini/Project2025/b.net.xml")  # Replace with actual network file path

    # ✅ Generate a random route
    route_edges = get_random_route(net)
    route_id = f"route_{self.vehicle_id}"  

    # ✅ Add route to SUMO
    if route_id not in traci.route.getIDList():
        traci.route.add(route_id, route_edges)

    # ✅ Add vehicle with the random route
    traci.vehicle.add(self.vehicle_id, route_id)

    return self._get_observation(), {}






def step(self, action):
    """Apply action and return observation, reward, done, info"""
    vehicle_id = "veh0"

    # ✅ Check if vehicle exists before changing target
    if vehicle_id in traci.vehicle.getIDList():
        traci.vehicle.changeTarget(vehicle_id, self.edge_list[action])
    else:
        print(f"🚨 Warning: Vehicle {vehicle_id} not found!")

    traci.simulationStep()

    # Example reward
    reward = -1  # Penalize long routes
    return self._get_observation(), reward, False, False, {}

