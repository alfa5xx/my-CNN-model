import os
# import gym
import numpy as np
from gym import spaces
import traci
import sumolib
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import Env

class SUMOPathEnv(gym.Env):
    """Custom Environment for optimal path finding in SUMO using SB3"""
    
    def __init__(self, sumocfg_file, start_edge, end_edge, gui=False):
        super(SUMOPathEnv, self).__init__()
        
        # SUMO configuration
        self.sumocfg_file = sumocfg_file
        self.sumo_binary = "sumo-gui" if gui else "sumo"
        self.start_edge = start_edge
        self.end_edge = end_edge
        
        # Load the SUMO network
        net_file = os.path.join(os.path.dirname(sumocfg_file), 
                               self._get_net_file_from_config(sumocfg_file))
        self.net = sumolib.net.readNet(net_file)
        
        # Vehicle information
        self.vehicle_id = "rl_vehicle"
        self.step_counter = 0
        self.max_steps = 1000
        self.traci_started = False
        self.current_route = []
        
        # Get all edges from the network
        self.all_edges = [edge.getID() for edge in self.net.getEdges()]
        self.edge_to_idx = {edge: i for i, edge in enumerate(self.all_edges)}
        
        # Define action and observation spaces
        # Actions: Choose from available next edges (simplified to 5 choices)
        self.action_space = spaces.Discrete(5)  # 5 possible directions/actions
        
        # Observations: Current edge, distance to target, traffic state
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0]),
            high=np.array([len(self.all_edges), 1, 1, 1, 1]),
            dtype=np.float32
        )
    
    def _get_net_file_from_config(self, sumocfg_file):
        """Extract the network file name from the SUMO config file"""
        import xml.etree.ElementTree as ET
        
        tree = ET.parse(sumocfg_file)
        root = tree.getroot()
        
        # Find the net-file element
        for input_elem in root.findall('.//input'):
            for net_file in input_elem.findall('.//net-file'):
                return net_file.get('value')
        
        # Default fallback
        return "network.xml"
    
    def _get_observation(self):
        """Get the current state observation"""
        if not traci.vehicle.getIDList() or self.vehicle_id not in traci.vehicle.getIDList():
            # Vehicle not in simulation yet
            return np.array([0, 1, 0, 0, 0], dtype=np.float32)
        
        current_edge = traci.vehicle.getRoadID(self.vehicle_id)
        
        # If vehicle is on a junction (edge ID starts with :), use previous edge
        if current_edge.startswith(':'):
            if hasattr(self, 'last_edge') and self.last_edge in self.edge_to_idx:
                edge_idx = self.edge_to_idx[self.last_edge]
            else:
                edge_idx = 0
        else:
            # Store last valid edge
            self.last_edge = current_edge
            edge_idx = self.edge_to_idx.get(current_edge, 0)
        
        # Normalize edge index
        norm_edge_idx = edge_idx / len(self.all_edges)
        
        # Calculate distance to destination (by route remaining)
        try:
            route = traci.vehicle.getRoute(self.vehicle_id)
            route_idx = traci.vehicle.getRouteIndex(self.vehicle_id)
            distance_to_end = 1.0 - (route_idx / len(route)) if len(route) > 0 else 1.0
        except:
            distance_to_end = 1.0
        
        # Get speed info
        try:
            max_speed = traci.vehicle.getAllowedSpeed(self.vehicle_id)
            current_speed = traci.vehicle.getSpeed(self.vehicle_id)
            norm_speed = current_speed / max_speed if max_speed > 0 else 0
        except:
            norm_speed = 0
        
        # Traffic density around (simple estimation)
        vehicle_count = len(traci.vehicle.getIDList())
        norm_traffic = min(1.0, vehicle_count / 50)  # Normalize traffic
        
        # Check if we're on a road with traffic lights
        tl_state = 0
        try:
            next_tls = traci.vehicle.getNextTLS(self.vehicle_id)
            if next_tls and len(next_tls) > 0:
                # 0 if green, 1 if red or yellow
                tl_state = 0 if next_tls[0][3] == 'G' else 1
        except:
            pass
            
        return np.array([
            norm_edge_idx,
            distance_to_end,
            norm_speed,
            norm_traffic,
            tl_state
        ], dtype=np.float32)
    
    def _get_next_edges(self, current_edge):
        """Get possible next edges from current position"""
        if current_edge.startswith(':'):
            return []
            
        possible_next = []
        
        # Get next edges from current route first
        try:
            route = traci.vehicle.getRoute(self.vehicle_id)
            route_idx = traci.vehicle.getRouteIndex(self.vehicle_id)
            if route_idx < len(route) - 1:
                possible_next.append(route[route_idx + 1])
        except:
            pass
        
        # Get all possible next edges from network
        edge_obj = self.net.getEdge(current_edge)
        for conn in edge_obj.getOutgoing():
            next_edge = conn.getID()
            if next_edge not in possible_next:
                possible_next.append(next_edge)
                
        return possible_next
    
    def _take_action(self, action):
        """Execute the selected action"""
        if not traci.vehicle.getIDList() or self.vehicle_id not in traci.vehicle.getIDList():
            return
        
        current_edge = traci.vehicle.getRoadID(self.vehicle_id)
        
        # Skip if on junction
        if current_edge.startswith(':'):
            return
            
        next_edges = self._get_next_edges(current_edge)
        if not next_edges:
            return
            
        # Map action to edge selection (simplified)
        selected_idx = min(action, len(next_edges) - 1)
        selected_edge = next_edges[selected_idx]
        
        # Create a new route from current position to target
        current_route = list(traci.vehicle.getRoute(self.vehicle_id))
        route_idx = traci.vehicle.getRouteIndex(self.vehicle_id)
        
        # Build new route: current part + selected edge + path to destination
        new_route = current_route[:route_idx+1]
        new_route.append(selected_edge)
        
        # Find path to destination
        try:
            from_edge = self.net.getEdge(selected_edge)
            to_edge = self.net.getEdge(self.end_edge)
            path, _ = self.net.getShortestPath(from_edge, to_edge)
            
            # Add remaining path
            for edge in path[1:]:  # Skip first as it's the selected edge
                new_route.append(edge.getID())
        except:
            # If path planning fails, just add the destination
            if self.end_edge not in new_route:
                new_route.append(self.end_edge)
        
        # Set the new route
        traci.vehicle.setRoute(self.vehicle_id, new_route)
    
    def step(self, action):
        """Take a step in the environment"""
        self._take_action(action)
        
        # Advance simulation
        traci.simulationStep()
        self.step_counter += 1
        
        # Get new observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if done
        terminated = self._is_done()
        truncated = False  # For time limits or other reasons to end episode
    
        # Info dictionary
        info = {}
        
        # Check if done
        done = self._is_done()
        
        # Info dictionary
        info = {}
        
        return  obs, reward, terminated, truncated, info
    
    def _calculate_reward(self):
        """Calculate reward based on progress and efficiency"""
        if not traci.vehicle.getIDList() or self.vehicle_id not in traci.vehicle.getIDList():
            return -10  # Big penalty if vehicle is not in simulation
        
        reward = 0
        
        # Check if reached destination
        current_edge = traci.vehicle.getRoadID(self.vehicle_id)
        if current_edge == self.end_edge:
            travel_time = self.step_counter
            # Higher reward for faster completion
            return 200 - (0.1 * travel_time)
            
        # Reward for making progress toward goal
        try:
            route = traci.vehicle.getRoute(self.vehicle_id)
            route_idx = traci.vehicle.getRouteIndex(self.vehicle_id)
            progress = route_idx / len(route) if len(route) > 0 else 0
            reward += progress * 5
        except:
            pass
        
        # Reward for maintaining speed
        try:
            max_speed = traci.vehicle.getAllowedSpeed(self.vehicle_id)
            current_speed = traci.vehicle.getSpeed(self.vehicle_id)
            speed_ratio = current_speed / max_speed if max_speed > 0 else 0
            reward += speed_ratio * 2
        except:
            pass
            
        # Small penalty for each step (encourages shorter paths)
        reward -= 0.1
        
        # Penalty for stopping
        if traci.vehicle.getSpeed(self.vehicle_id) < 0.1:
            reward -= 0.5
            
        return reward
    
    def _is_done(self):
        """Check if the episode is finished"""
        # Episode is done if:
        # 1. Max steps reached
        # 2. Vehicle reached destination
        # 3. Vehicle is no longer in simulation
        
        if self.step_counter >= self.max_steps:
            return True
            
        if not traci.vehicle.getIDList() or self.vehicle_id not in traci.vehicle.getIDList():
            return True
            
        current_edge = traci.vehicle.getRoadID(self.vehicle_id)
        if current_edge == self.end_edge:
            return True
            
        return False
    
    # def reset(self, seed=None, options=None):
     
        
    #  def reset(self, seed=None, options=None):
    # """Reset the environment for a new episode"""
    # # If seed is provided, you can use it for reproducibility
    #     if seed is not None:
    #         np.random.seed(seed)
            
    #     # Close existing connection if any
    #     if self.traci_started:
    #         traci.close()
        
    #     # Start SUMO
    #     sumo_cmd = [self.sumo_binary, 
    #                 "-c", self.sumocfg_file,
    #                 "--no-warnings", "true"]
    #     traci.start(sumo_cmd)
    #     self.traci_started = True
        
    #     # Reset counters
    #     self.step_counter = 0
    
    #     # Create a route from start to end
    #     try:
    #         # Find shortest path
    #         from_edge = self.net.getEdge(self.start_edge)
    #         to_edge = self.net.getEdge(self.end_edge)
    #         path, _ = self.net.getShortestPath(from_edge, to_edge)
            
    #         # Convert to IDs
    #         route = [edge.getID() for edge in path]
            
    #         # Add route to SUMO
    #         route_id = "route_" + self.vehicle_id
    #         traci.route.add(route_id, route)
            
    #         # Add vehicle - using DEFAULT_VEHTYPE instead of passenger
    #         traci.vehicle.add(
    #             self.vehicle_id,
    #             route_id,
    #             typeID="DEFAULT_VEHTYPE",  # Use a vehicle type that exists in your SUMO config
    #             depart=0
    #         )
            
    #         # Set vehicle parameters
    #         traci.vehicle.setColor(self.vehicle_id, (255, 0, 0, 255))  # Red color for visibility
    #         traci.vehicle.setSpeedFactor(self.vehicle_id, 1.0)  # Normal speed
    #     except Exception as e:
    #         print(f"Error creating route: {e}")
        
    #     # Run one step to initialize
    #     traci.simulationStep()
        
    #     # Create info dictionary
    #     info = {}
        
    #     # Return initial observation and info
    # return self._get_observation(), info
     
    def reset(self, seed=None, options=None):
        "Reset the environment for a new episode" 
    
    # If seed is provided, you can use it for reproducibility
        if seed is not None:
            np.random.seed(seed)
        
        # Close existing connection if any
        if self.traci_started:
            traci.close()
        
        # Start SUMO
        sumo_cmd = [self.sumo_binary, 
                    "-c", self.sumocfg_file,
                    "--no-warnings", "true"]
        traci.start(sumo_cmd)
        self.traci_started = True
        
        # Reset counters
        self.step_counter = 0
    
        # Create a route from start to end
        try:
            # Find shortest path
            from_edge = self.net.getEdge(self.start_edge)
            to_edge = self.net.getEdge(self.end_edge)
            path, _ = self.net.getShortestPath(from_edge, to_edge)
            
            # Convert to IDs
            route = [edge.getID() for edge in path]
            
            # Add route to SUMO
            route_id = "route_" + self.vehicle_id
            traci.route.add(route_id, route)
            
            # Add vehicle - using DEFAULT_VEHTYPE instead of passenger
            traci.vehicle.add(
                self.vehicle_id,
                route_id,
                typeID="DEFAULT_VEHTYPE",  # Use a vehicle type that exists in your SUMO config
                depart=0
            )
            
            # Set vehicle parameters
            traci.vehicle.setColor(self.vehicle_id, (255, 0, 0, 255))  # Red color for visibility
            traci.vehicle.setSpeedFactor(self.vehicle_id, 1.0)  # Normal speed
        
        except Exception as e:
            print(f"Error creating route: {e}")
        
        # Run one step to initialize
        traci.simulationStep()
        
        # Create info dictionary
        info = {}

    # Return initial observation and info
        return self._get_observation(), info
    
    
    
    
        """Reset the environment for a new episode"""
        # Close existing connection if any
        if seed is not None:
            np.random.seed(seed)
            
        # Close existing connection if any
        if self.traci_started:
            traci.close()
            
       
        
        # Start SUMO
        sumo_cmd = [self.sumo_binary, 
                    "-c", self.sumocfg_file,
                    "--no-warnings", "true"]
        traci.start(sumo_cmd)
        self.traci_started = True
        
        # Reset counters
        self.step_counter = 0
        
        # Create a route from start to end
        try:
            # Find shortest path
            from_edge = self.net.getEdge(self.start_edge)
            to_edge = self.net.getEdge(self.end_edge)
            path, _ = self.net.getShortestPath(from_edge, to_edge)
            
            # Convert to IDs
            route = [edge.getID() for edge in path]
            
            # Add route to SUMO
            route_id = "route_" + self.vehicle_id
            traci.route.add(route_id, route)
            
            # Add vehicle
            # traci.vehicle.add(
            #     self.vehicle_id,
            #     route_id,
            #     typeID="passenger",
            #     depart=0
            # )
            
            traci.vehicle.add(
            self.vehicle_id,
            route_id,
            typeID="DEFAULT_VEHTYPE",  # Using default vehicle type
            depart=0
            )
            
            
            # Set vehicle parameters
            traci.vehicle.setColor(self.vehicle_id, (255, 0, 0, 255))  # Red color for visibility
            traci.vehicle.setSpeedFactor(self.vehicle_id, 1.0)  # Normal speed
        except Exception as e:
            print(f"Error creating route: {e}")
        
        # Run one step to initialize
        traci.simulationStep()
        
        
        # Create info dictionary first
        info = {}
        
        # Return initial observation
        return self._get_observation() ,info
    
    def render(self, mode='human'):
        """Render the environment (not needed when using SUMO-GUI)"""
        pass
    
    def close(self):
        """Close the environment and SUMO connection"""
        if self.traci_started:
            traci.close()
            self.traci_started = False


def train_agent(sumocfg_file, start_edge, end_edge, iterations=10000):
    """Train the SB3 agent to find optimal paths"""
    # Create output directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Create and wrap the environment
    env = SUMOPathEnv(sumocfg_file, start_edge, end_edge, gui=False)
    env = Monitor(env, "C:/Users/Administrator/Desktop/Prof.Mangini/Project2025/logs/")
    env = DummyVecEnv([lambda: env])
    
    # Create the PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        tensorboard_log="logs/tensorboard/"
    )
    
    # Set up checkpointing
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path="models/",
        name_prefix="ppo_sumo_pathfinder"
    )
    
    # Train the model
    model.learn(
        total_timesteps=iterations,
        callback=checkpoint_callback
    )
    
    # Save the final model
    model.save("models/ppo_sumo_pathfinder_final")
    
    print(f"Training completed after {iterations} iterations")
    return model


def evaluate_agent(model, sumocfg_file, start_edge, end_edge, episodes=5):
    """Evaluate the trained agent with visualization"""
    # Create environment with GUI
    env = SUMOPathEnv(sumocfg_file, start_edge, end_edge, gui=True)
    
    total_reward = 0
    total_steps = 0
    
    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        ep_steps = 0
        
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Execute action
            obs, reward, done, _ = env.step(action)
            
            ep_reward += reward
            ep_steps += 1
        
        print(f"Episode {ep+1}: Reward = {ep_reward}, Steps = {ep_steps}")
        total_reward += ep_reward
        total_steps += ep_steps
    
    print(f"Average Reward: {total_reward/episodes}")
    print(f"Average Steps: {total_steps/episodes}")
    
    env.close()


def compare_with_shortest_path(sumocfg_file, start_edge, end_edge, episodes=5):
    """Compare RL agent with traditional shortest path"""
    # Create environments - one with RL control, one with shortest path
    env_rl = SUMOPathEnv(sumocfg_file, start_edge, end_edge, gui=True)
    
    # Load the trained model
    model = PPO.load("models/ppo_sumo_pathfinder_final")
    
    # Compare metrics
    rl_times = []
    rl_distances = []
    
    for ep in range(episodes):
        # Evaluate RL agent
        obs = env_rl.reset()
        done = False
        rl_steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env_rl.step(action)
            rl_steps += 1
        
        # Calculate travel distance
        try:
            rl_distance = traci.vehicle.getDistance(env_rl.vehicle_id)
        except:
            rl_distance = 0
            
        rl_times.append(rl_steps)
        rl_distances.append(rl_distance)
        
        print(f"Episode {ep+1}: RL Time = {rl_steps}, RL Distance = {rl_distance:.2f}")
    
    env_rl.close()
    
    # Run traditional shortest path for comparison
    env_sp = SUMOPathEnv(sumocfg_file, start_edge, end_edge, gui=True)
    sp_times = []
    sp_distances = []
    
    for ep in range(episodes):
        obs = env_sp.reset()
        done = False
        sp_steps = 0
        
        # Just run the simulation without RL actions (using initial shortest path)
        while not done:
            obs, _, done, _ = env_sp.step(0)  # Always choose default path
            sp_steps += 1
        
        # Calculate travel distance
        try:
            sp_distance = traci.vehicle.getDistance(env_sp.vehicle_id)
        except:
            sp_distance = 0
            
        sp_times.append(sp_steps)
        sp_distances.append(sp_distance)
        
        print(f"Episode {ep+1}: SP Time = {sp_steps}, SP Distance = {sp_distance:.2f}")
    
    env_sp.close()
    
    # Display results
    print("\nResults Comparison:")
    print(f"RL Agent - Avg Time: {sum(rl_times)/len(rl_times):.2f}, Avg Distance: {sum(rl_distances)/len(rl_distances):.2f}")
    print(f"Shortest Path - Avg Time: {sum(sp_times)/len(sp_times):.2f}, Avg Distance: {sum(sp_distances)/len(sp_distances):.2f}")
    
    # Calculate improvement
    time_improvement = (sum(sp_times) - sum(rl_times)) / sum(sp_times) * 100
    print(f"Time Improvement: {time_improvement:.2f}%")


if __name__ == "__main__":
    # Replace these with your actual file paths and edge IDs
    sumocfg_file = "C:/Users/Administrator/Desktop/Prof.Mangini/Project2025/b.sumocfg"
    start_edge = "E4"  # Replace with actual edge ID
    end_edge = "E10"      # Replace with actual edge ID
    
    # 1. Train the agent (increase iterations for better performance)
    print("Starting training...")
    model = train_agent(sumocfg_file, start_edge, end_edge, iterations=10000)
    
    # 2. Evaluate the trained agent
    print("\nEvaluating trained agent...")
    evaluate_agent(model, sumocfg_file, start_edge, end_edge)
    
    # 3. Compare with traditional shortest path
    print("\nComparing with traditional shortest path...")
    compare_with_shortest_path(sumocfg_file, start_edge, end_edge)