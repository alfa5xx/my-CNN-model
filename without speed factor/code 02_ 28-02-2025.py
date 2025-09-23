import os
import gym
import numpy as np
from gym import spaces
import traci
import sumolib
import subprocess
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

class SUMOEnvironment(gym.Env):
    """Custom Environment that follows gym interface for SUMO path optimization"""
    metadata = {'render.modes': ['human']}

    def __init__(self, sumocfg_file, start_edge, end_edge, gui=False):
        super(SUMOEnvironment, self).__init__()
        
        # SUMO configuration
        self.sumocfg_file = sumocfg_file
        self.sumo_binary = "sumo-gui" if gui else "sumo"
        self.start_edge = start_edge
        self.end_edge = end_edge
        
        # Load SUMO network
        self.net = sumolib.net.readNet(os.path.dirname(sumocfg_file) + "C:/Users/Administrator/Desktop/Prof.Mangini/Project2025/b.net.xml")
        
        # Get all edges in the network
        self.all_edges = [edge.getID() for edge in self.net.getEdges()]
        self.edge_count = len(self.all_edges)
        
        # Define action and observation space
        # Actions: Choose the next edge to move to
        self.action_space = spaces.Discrete(5)  # Move in 5 directions: forward, left, right, stay, u-turn
        
        # Observations: Current edge, distance to target, traffic density around, etc.
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([self.edge_count, 1, 1, 1]),
            dtype=np.float32
        )
        
        # Initialize simulation
        self.vehicle_id = "rl_vehicle"
        self.current_step = 0
        self.max_steps = 1000  # Maximum steps per episode
        self.traci_started = False
    
    def _get_state(self):
        """Get the current state (observation) of the environment"""
        if not traci.vehicle.getIDList() or self.vehicle_id not in traci.vehicle.getIDList():
            # Vehicle not yet inserted or removed
            normalized_edge_idx = 0
            normalized_distance = 1.0  # Maximum distance
            normalized_speed = 0.0  # Stopped
            normalized_traffic = 0.5  # Medium traffic (placeholder)
        else:
            # Get current edge index
            current_edge = traci.vehicle.getRoadID(self.vehicle_id)
            if current_edge.startswith(":"):  # Junction
                normalized_edge_idx = 0  # Placeholder for junctions
            else:
                edge_idx = self.all_edges.index(current_edge) if current_edge in self.all_edges else 0
                normalized_edge_idx = edge_idx / self.edge_count
            
            # Get distance to target (normalized)
            route = traci.vehicle.getRoute(self.vehicle_id)
            route_idx = traci.vehicle.getRouteIndex(self.vehicle_id)
            remaining_route = route[route_idx:]
            normalized_distance = len(remaining_route) / len(route)
            
            # Get current speed (normalized)
            max_speed = traci.vehicle.getAllowedSpeed(self.vehicle_id)
            current_speed = traci.vehicle.getSpeed(self.vehicle_id)
            normalized_speed = current_speed / max_speed if max_speed > 0 else 0
            
            # Get traffic density (placeholder - you would implement actual traffic sensing)
            normalized_traffic = 0.5  # Medium traffic
        
        return np.array([
            normalized_edge_idx,
            normalized_distance,
            normalized_speed,
            normalized_traffic
        ], dtype=np.float32)
    
    def _take_action(self, action):
        """Execute the action in the SUMO environment"""
        if not traci.vehicle.getIDList() or self.vehicle_id not in traci.vehicle.getIDList():
            return  # Vehicle not yet inserted or removed
        
        current_edge = traci.vehicle.getRoadID(self.vehicle_id)
        if current_edge.startswith(":"):  # Vehicle is on a junction
            return  # Wait until vehicle is on a regular edge
            
        # Get next possible edges
        next_edges = []
        try:
            current_route = traci.vehicle.getRoute(self.vehicle_id)
            route_idx = traci.vehicle.getRouteIndex(self.vehicle_id)
            if route_idx < len(current_route) - 1:
                next_planned_edge = current_route[route_idx + 1]
                next_edges.append(next_planned_edge)
        except:
            pass
            
        # Get all possible next edges from current edge
        if current_edge in self.all_edges:
            edge_obj = self.net.getEdge(current_edge)
            for connection in edge_obj.getOutgoing():
                next_edge = connection.getID()
                if next_edge not in next_edges:
                    next_edges.append(next_edge)
        
        if len(next_edges) == 0:
            return  # No next edges available
            
        # Map action to edge selection
        if action == 0:  # Forward (keep current plan if possible)
            selected_edge = next_edges[0] if next_edges else None
        elif action == 1 and len(next_edges) > 1:  # Left
            selected_edge = next_edges[1]
        elif action == 2 and len(next_edges) > 2:  # Right
            selected_edge = next_edges[2]
        elif action == 3:  # Stay (slow down)
            traci.vehicle.slowDown(self.vehicle_id, 1.0, 3)
            return
        elif action == 4 and len(next_edges) > 3:  # U-turn if available
            selected_edge = next_edges[3]
        else:
            selected_edge = next_edges[0] if next_edges else None
            
        # Update route if needed
        if selected_edge:
            current_route = list(traci.vehicle.getRoute(self.vehicle_id))
            route_idx = traci.vehicle.getRouteIndex(self.vehicle_id)
            
            # Create new route
            new_route = current_route[:route_idx+1]
            new_route.append(selected_edge)
            
            # Find path to destination
            try:
                dest_path = self.net.getShortestPath(
                    self.net.getEdge(selected_edge),
                    self.net.getEdge(self.end_edge)
                )[0]
                
                # Add path to destination
                for edge in dest_path:
                    if edge.getID() not in new_route:
                        new_route.append(edge.getID())
            except:
                # If no path found, keep original route
                new_route = current_route
                
            # Set new route
            traci.vehicle.setRoute(self.vehicle_id, new_route)
            
    def step(self, action):
        """Execute one time step within the environment"""
        self._take_action(action)
        
        # Advance simulation
        traci.simulationStep()
        self.current_step += 1
        
        # Get new state
        state = self._get_state()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self._is_done()
        
        return state, reward, done, {}
    
    def _calculate_reward(self):
        """Calculate the reward based on current state"""
        reward = 0
        
        if not traci.vehicle.getIDList() or self.vehicle_id not in traci.vehicle.getIDList():
            return -10  # Vehicle not in simulation (big penalty)
        
        # Check if reached destination
        current_edge = traci.vehicle.getRoadID(self.vehicle_id)
        if current_edge == self.end_edge:
            return 100  # Big reward for reaching destination
        
        # Reward for making progress
        try:
            route = traci.vehicle.getRoute(self.vehicle_id)
            route_idx = traci.vehicle.getRouteIndex(self.vehicle_id)
            route_progress = route_idx / len(route) if len(route) > 0 else 0
            reward += route_progress * 10
        except:
            pass
        
        # Penalty for time
        reward -= 0.1  # Small penalty for each step
        
        # Reward for speed
        try:
            max_speed = traci.vehicle.getAllowedSpeed(self.vehicle_id)
            current_speed = traci.vehicle.getSpeed(self.vehicle_id)
            speed_ratio = current_speed / max_speed if max_speed > 0 else 0
            reward += speed_ratio * 0.5
        except:
            pass
            
        return reward
    
    def _is_done(self):
        """Check if episode is done"""
        # Episode is done if:
        # 1. Vehicle reached destination
        # 2. Max steps reached
        # 3. Vehicle is no longer in simulation
        
        if self.current_step >= self.max_steps:
            return True
            
        if not traci.vehicle.getIDList() or self.vehicle_id not in traci.vehicle.getIDList():
            return True
            
        current_edge = traci.vehicle.getRoadID(self.vehicle_id)
        if current_edge == self.end_edge:
            return True
            
        return False
    
    def reset(self):
        """Reset the state of the environment to an initial state"""
        # Close existing TRACI connection if any
        if self.traci_started:
            traci.close()
            self.traci_started = False
        
        # Start SUMO simulation
        sumo_cmd = [self.sumo_binary, "-c", self.sumocfg_file]
        traci.start(sumo_cmd)
        self.traci_started = True
        
        # Reset step counter
        self.current_step = 0
        
        # Insert vehicle with a route from start to end
        try:
            source_edge = self.net.getEdge(self.start_edge)
            target_edge = self.net.getEdge(self.end_edge)
            path_edges, _ = self.net.getShortestPath(source_edge, target_edge)
            
            route_id = "route_" + self.vehicle_id
            path_ids = [edge.getID() for edge in path_edges]
            
            traci.route.add(route_id, path_ids)
            traci.vehicle.add(
                self.vehicle_id, 
                route_id,
                typeID="passenger",
                depart=0
            )
        except Exception as e:
            print(f"Error inserting vehicle: {e}")
        
        # Step once to initialize
        traci.simulationStep()
        
        # Return initial observation
        return self._get_state()
    
    def render(self, mode='human'):
        """Render the environment (only relevant when using sumo-gui)"""
        pass
    
    def close(self):
        """Close the environment"""
        if self.traci_started:
            traci.close()
            self.traci_started = False

def train_rl_agent(sumocfg_file, start_edge, end_edge, total_timesteps=10000):
    """Train a RL agent for path finding in SUMO"""
    # Create log directory
    log_dir = "C:/Users/Administrator/Desktop/Prof.Mangini/Project2025/logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create and wrap the environment
    env = SUMOEnvironment(sumocfg_file, start_edge, end_edge, gui=False)
    env = Monitor(env, log_dir)
    env = DummyVecEnv([lambda: env])
    
    # Create the RL model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=log_dir
    )
    
    # Set up checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path="./models/",
        name_prefix="ppo_sumo"
    )
    
    # Train the agent
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback
    )
    
    # Save the final model
    model.save("models/ppo_sumo_final")
    
    return model

def evaluate_agent(model, sumocfg_file, start_edge, end_edge, num_episodes=10):
    """Evaluate a trained agent on the SUMO environment"""
    env = SUMOEnvironment(sumocfg_file, start_edge, end_edge, gui=True)
    
    total_rewards = 0
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
        
        print(f"Episode {episode+1}: Reward = {episode_reward}")
        total_rewards += episode_reward
    
    print(f"Average Reward over {num_episodes} episodes: {total_rewards/num_episodes}")
    env.close()

if __name__ == "__main__":
    # File paths
    sumocfg_file = "C:/Users/Administrator/Desktop/Prof.Mangini/Project2025/b.sumocfg/"
    
    # Define start and end points
    start_edge = "E4"  # Replace with actual edge ID
    end_edge = "E10"      # Replace with actual edge ID
    
    # Train the agent
    model = train_rl_agent(sumocfg_file, start_edge, end_edge, total_timesteps=10000)
    
    # Evaluate the agent
    evaluate_agent(model, sumocfg_file, start_edge, end_edge)