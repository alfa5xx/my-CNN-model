import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import tensorflow as tf
from tensorflow.keras import layers, models
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import random
import traci
import sumolib
import matplotlib.pyplot as plt
from collections import defaultdict

# Add SUMO_HOME to the Python path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

class SUMOPathFinderEnv(gym.Env):
    """
    Custom SUMO environment for path finding using RL
    """
    def __init__(self, sumocfg_file, net_file, route_file, render_mode=None):
        super(SUMOPathFinderEnv, self).__init__()
        
        # Store SUMO configuration files
        self.sumocfg_file = sumocfg_file
        self.net_file = net_file
        self.route_file = route_file
        
        # Load SUMO network
        self.net = sumolib.net.readNet(net_file)
        self.edges = [edge.getID() for edge in self.net.getEdges()]
        self.edge_dict = {edge: i for i, edge in enumerate(self.edges)}
        self.num_edges = len(self.edges)
        self.traffic_lights = self.net.getTrafficLights()
        
        # Simulation parameters
        self.sumo_cmd = ["sumo", "-c", sumocfg_file]
        self.step_length = 1.0  # seconds
        self.max_steps = 1000
        self.current_step = 0
        
        # Environment state
        self.start_edge = None
        self.end_edge = None
        self.current_edge = None
        self.visited_edges = set()
        
        # State representation: current edge, end edge, traffic, and distance features
        # CNN input shape: 3 channels (traffic, distance, traffic lights)
        self.grid_size = int(np.ceil(np.sqrt(self.num_edges)))
        
        # Action space: select next edge from available connections
        self.action_space = spaces.Discrete(10)  # Max number of connections from any edge
        
        # Observation space: Grid representation for CNN + current and target edge indices
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size, 3), dtype=np.float32),
            'current_pos': spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
            'target_pos': spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
        })
        
        # Path tracking
        self.current_path = []
        self.best_path = []
        self.best_time = float('inf')
        
        # CNN model for feature extraction
        self.cnn_model = self._build_cnn_model()
    
    def _build_cnn_model(self):
        """Build CNN model for feature extraction"""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                         input_shape=(self.grid_size, self.grid_size, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
    
    # Close any existing SUMO connection
        try:
            traci.close()
        except:
            pass  # Connection might not exist, so we catch the exception
    
        # Start SUMO simulation
        traci.start(self.sumo_cmd)
    
        # Reset simulation parameters
        self.current_step = 0
        self.visited_edges = set()
        self.current_path = []
    
        # Randomly select start and end edges
        valid_edges = [e for e in self.edges if len(self.net.getEdge(e).getOutgoing()) > 0 and e.startswith(':') is False]
        
        if not options or 'start_edge' not in options or 'end_edge' not in options:
            self.start_edge = random.choice(valid_edges)
            remaining_edges = [e for e in valid_edges if e != self.start_edge]
            self.end_edge = random.choice(remaining_edges)
        else:
            self.start_edge = options['start_edge']
            self.end_edge = options['end_edge']
        
        # Set current edge to start edge
        self.current_edge = self.start_edge
        self.visited_edges.add(self.current_edge)
        self.current_path.append(self.current_edge)
        
        # Add a vehicle to the simulation at the start edge
        veh_id = "rl_vehicle"
        try:
            traci.vehicle.add(veh_id, "rl_route", typeID="passenger")
            traci.vehicle.moveToXY(veh_id, self.start_edge, 0, 
                                  self.net.getEdge(self.start_edge).getFromNode().getCoord()[0],
                                  self.net.getEdge(self.start_edge).getFromNode().getCoord()[1], angle=0, keepRoute=2)
        except Exception as e:
            print(f"Warning: Could not add vehicle: {e}")
        
        # Get initial observation
        observation = self._get_observation()
        
        return observation
    
def _get_observation(self):
        """Get the current observation"""
        # Create grid representation
        grid = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        
        # Fill grid with traffic information, distance to target, and traffic light info
        for i, edge_id in enumerate(self.edges):
            if edge_id.startswith(':'):  # Skip internal edges
                continue
                
            # Convert edge index to 2D position
            row = i // self.grid_size
            col = i % self.grid_size
            
            if row >= self.grid_size or col >= self.grid_size:
                continue
            
            # Channel 0: Traffic density
            if traci.isConnected():
                try:
                    vehicle_count = traci.edge.getLastStepVehicleNumber(edge_id)
                    max_vehicles = max(1, traci.edge.getLastStepVehicleNumber(edge_id))
                    grid[row, col, 0] = min(1.0, vehicle_count / max_vehicles)
                except traci.exceptions.TraCIException:
                    grid[row, col, 0] = 0.0
            
            # Channel 1: Distance to target (normalized)
            try:
                if self.end_edge:
                    source_node = self.net.getEdge(edge_id).getFromNode()
                    target_node = self.net.getEdge(self.end_edge).getToNode()
                    distance = np.sqrt(
                        (source_node.getCoord()[0] - target_node.getCoord()[0]) ** 2 +
                        (source_node.getCoord()[1] - target_node.getCoord()[1]) ** 2
                    )
                    # Normalize distance
                    max_dist = 5000  # Assuming max distance is 5000 meters
                    grid[row, col, 1] = min(1.0, distance / max_dist)
            except:
                grid[row, col, 1] = 1.0  # Max distance if can't calculate
            
            # Channel 2: Traffic light information
            edge = self.net.getEdge(edge_id)
            has_traffic_light = 0.0
            for connection in edge.getOutgoing():
                if connection.getTLSID() is not None:
                    has_traffic_light = 1.0
                    break
            grid[row, col, 2] = has_traffic_light
        
        # Current position and target position as normalized coordinates
        current_idx = self.edge_dict.get(self.current_edge, 0)
        current_row = (current_idx // self.grid_size) / self.grid_size
        current_col = (current_idx % self.grid_size) / self.grid_size
        
        target_idx = self.edge_dict.get(self.end_edge, 0)
        target_row = (target_idx // self.grid_size) / self.grid_size
        target_col = (target_idx % self.grid_size) / self.grid_size
        
        return {
            'grid': grid,
            'current_pos': np.array([current_row, current_col], dtype=np.float32),
            'target_pos': np.array([target_row, target_col], dtype=np.float32),
        }
    
def step(self, action):
        """Take a step in the environment"""
        self.current_step += 1
        
        # Get available connections from current edge
        current_edge_obj = self.net.getEdge(self.current_edge)
        available_connections = []
        
        for conn in current_edge_obj.getOutgoing():
            target_edge = conn.getToEdge().getID()
            if not target_edge.startswith(':'):  # Skip internal edges
                available_connections.append(target_edge)
        
        # Default action is to stay if there are no connections or invalid action
        if not available_connections or action >= len(available_connections):
            next_edge = self.current_edge
            reward = -10  # Penalize invalid actions
        else:
            next_edge = available_connections[action]
        
        # Update current edge and path
        self.current_edge = next_edge
        self.visited_edges.add(next_edge)
        self.current_path.append(next_edge)
        
        # Calculate reward
        reward = self._calculate_reward(next_edge)
        
        # Check if we've reached the goal
        done = (next_edge == self.end_edge) or (self.current_step >= self.max_steps)
        
        # Get next observation
        observation = self._get_observation()
        
        # Run one simulation step
        if traci.isConnected():
            traci.simulationStep()
        
        # Update best path if we've reached the goal
        if next_edge == self.end_edge:
            travel_time = self._calculate_travel_time(self.current_path)
            if travel_time < self.best_time:
                self.best_time = travel_time
                self.best_path = self.current_path.copy()
        
        # Return observation, reward, done, truncated, info
        info = {
            'path': self.current_path,
            'travel_time': self._calculate_travel_time(self.current_path),
            'reached_goal': next_edge == self.end_edge
        }
        
        return observation, reward, done, False, info
    
def _calculate_reward(self, edge):
        """Calculate reward for the current step"""
        if edge == self.end_edge:
            return 100  # Reached the goal
        
        # Penalty for revisiting edges
        if self.current_path.count(edge) > 1:
            return -5
        
        # Calculate distance to goal
        current_node = self.net.getEdge(edge).getToNode()
        target_node = self.net.getEdge(self.end_edge).getToNode()
        
        distance = np.sqrt(
            (current_node.getCoord()[0] - target_node.getCoord()[0]) ** 2 +
            (current_node.getCoord()[1] - target_node.getCoord()[1]) ** 2
        )
        
        # Normalize distance (assuming max distance is 5000 meters)
        norm_distance = min(1.0, distance / 5000)
        
        # Reward for getting closer to the goal
        proximity_reward = -norm_distance * 10
        
        # Penalty for edges with traffic lights
        tl_penalty = 0
        edge_obj = self.net.getEdge(edge)
        for connection in edge_obj.getOutgoing():
            if connection.getTLSID() is not None:
                tl_penalty -= 2
                break
        
        # Penalty for edges with high traffic
        traffic_penalty = 0
        if traci.isConnected():
            try:
                vehicle_count = traci.edge.getLastStepVehicleNumber(edge)
                traffic_penalty = -vehicle_count
            except:
                pass
        
        # Penalty for time
        time_penalty = -1
        
        return proximity_reward + tl_penalty + traffic_penalty + time_penalty
    
def _calculate_travel_time(self, path):
        """Calculate the travel time for a given path"""
        if not path:
            return float('inf')
        
        total_time = 0
        for i in range(len(path) - 1):
            edge1 = path[i]
            edge2 = path[i + 1]
            
            # Get travel time for edge
            edge_obj = self.net.getEdge(edge1)
            length = edge_obj.getLength()
            speed = edge_obj.getSpeed()
            
            if speed > 0:
                time = length / speed
            else:
                time = length / 13.89  # Default 50 km/h in m/s
            
            # Add delay if there's a traffic light
            for connection in edge_obj.getOutgoing():
                if connection.getToEdge().getID() == edge2 and connection.getTLSID() is not None:
                    time += 15  # Average traffic light delay
            
            # Add traffic delay
            if traci.isConnected():
                try:
                    vehicle_count = traci.edge.getLastStepVehicleNumber(edge1)
                    time += vehicle_count * 2  # 2 seconds per vehicle
                except:
                    pass
            
            total_time += time
        
        return total_time
    
def close(self):
        """Close the environment"""
        if traci.isConnected():
            traci.close()
    
def render(self):
        """Render the environment (not implemented)"""
        pass

class SaveBestPathCallback(BaseCallback):
    """Callback to save the best path during training"""
    def __init__(self, verbose=0):
        super(SaveBestPathCallback, self).__init__(verbose)
        self.best_paths = []
        self.best_rewards = []
    
    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        if info['reached_goal']:
            self.best_paths.append(info['path'])
            self.best_rewards.append(self.locals['rewards'][0])
        return True

class CNNFeatureExtractor:
    """Feature extractor class for CNN preprocessing"""
    def __init__(self, env):
        self.env = env
        self.cnn_model = env.cnn_model
    
    def extract_features(self, observation):
        """Extract features from observation using CNN"""
        grid = np.expand_dims(observation['grid'], axis=0)
        features = self.cnn_model.predict(grid, verbose=0)[0]
        
        # Combine with current and target position
        combined_features = np.concatenate([
            features,
            observation['current_pos'],
            observation['target_pos']
        ])
        
        return combined_features

def wrap_env_with_cnn(env):
    """Wrap environment with CNN feature extractor"""
    feature_extractor = CNNFeatureExtractor(env)
    
    class CNNEnvWrapper(gym.Wrapper):
        def __init__(self, env):
            super(CNNEnvWrapper, self).__init__(env)
            self.feature_extractor = feature_extractor
            # Update observation space for the wrapped environment
            feature_dim = 64 + 4  # CNN output + current/target positions
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(feature_dim,), dtype=np.float32
            )
        
        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            cnn_features = self.feature_extractor.extract_features(obs)
            return cnn_features, info
        
        def step(self, action):
            obs, reward, done, truncated, info = self.env.step(action)
            cnn_features = self.feature_extractor.extract_features(obs)
            return cnn_features, reward, done, truncated, info
    
    return CNNEnvWrapper(env)

def train_agent(env, total_timesteps=100000):
    """Train a RL agent using PPO from Stable Baselines 3"""
    # Wrap environment with CNN feature extractor
    wrapped_env = wrap_env_with_cnn(env)
    
    # Create callback to track best paths
    callback = SaveBestPathCallback()
    
    # Create and train PPO agent
    model = PPO("MlpPolicy", wrapped_env, verbose=1, 
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01)
    
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    return model, callback.best_paths, callback.best_rewards

def visualize_path(net_file, path):
    """Visualize the path on the network"""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D
    
    # Load network
    net = sumolib.net.readNet(net_file)
    
    plt.figure(figsize=(12, 8))
    
    # Plot all edges in light gray
    for edge in net.getEdges():
        if edge.getID().startswith(':'):  # Skip internal edges
            continue
        
        shape = edge.getShape()
        x = [p[0] for p in shape]
        y = [p[1] for p in shape]
        
        plt.plot(x, y, color='lightgray', linewidth=1, zorder=1)
    
    # Plot path edges in blue with gradient (darker blue for later edges)
    cmap = plt.cm.Blues
    for i in range(len(path) - 1):
        edge_id = path[i]
        edge = net.getEdge(edge_id)
        
        shape = edge.getShape()
        x = [p[0] for p in shape]
        y = [p[1] for p in shape]
        
        color = cmap(0.5 + 0.5 * (i / len(path)))
        plt.plot(x, y, color=color, linewidth=3, zorder=2)
    
    # Mark start and end points
    start_edge = net.getEdge(path[0])
    end_edge = net.getEdge(path[-1])
    
    start_point = start_edge.getFromNode().getCoord()
    end_point = end_edge.getToNode().getCoord()
    
    plt.scatter(start_point[0], start_point[1], color='green', s=100, zorder=3, label='Start')
    plt.scatter(end_point[0], end_point[1], color='red', s=100, zorder=3, label='End')
    
    # Add legend
    plt.legend()
    plt.title(f"Optimal Path (Length: {len(path)} edges)")
    plt.axis('equal')
    plt.grid(True)
    
    # Save plot
    plt.savefig("optimal_path.png", dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_path(env, path):
    """Evaluate a path based on different metrics"""
    # Calculate travel time
    travel_time = env._calculate_travel_time(path)
    
    # Count traffic lights
    traffic_light_count = 0
    for i in range(len(path) - 1):
        edge1 = path[i]
        edge2 = path[i + 1]
        
        edge_obj = env.net.getEdge(edge1)
        for connection in edge_obj.getOutgoing():
            if connection.getToEdge().getID() == edge2 and connection.getTLSID() is not None:
                traffic_light_count += 1
    
    # Calculate total distance
    total_distance = 0
    for edge_id in path:
        edge_obj = env.net.getEdge(edge_id)
        total_distance += edge_obj.getLength()
    
    # Count vehicles per edge
    vehicle_counts = []
    if traci.isConnected():
        for edge_id in path:
            try:
                vehicle_count = traci.edge.getLastStepVehicleNumber(edge_id)
                vehicle_counts.append(vehicle_count)
            except:
                vehicle_counts.append(0)
    
    return {
        'travel_time': travel_time,
        'traffic_light_count': traffic_light_count,
        'total_distance': total_distance,
        'path_length': len(path),
        'vehicle_counts': vehicle_counts,
        'avg_vehicles_per_edge': sum(vehicle_counts) / len(vehicle_counts) if vehicle_counts else 0
    }

def find_optimal_path(sumocfg_file, net_file, route_file, start_edge=None, end_edge=None, 
                      training_steps=50000, visualize=True):
    """Main function to find optimal path between two edges"""
    # Create environment
    env = SUMOPathFinderEnv(sumocfg_file, net_file, route_file)
    
    # Train agent
    print("Training agent...")
    model, best_paths, best_rewards = train_agent(env, total_timesteps=training_steps)
    
    # Find best path from training
    if best_paths:
        best_path_idx = np.argmax(best_rewards)
        best_path = best_paths[best_path_idx]
    else:
        # If no successful paths during training, test the trained model
        print("No successful paths during training. Testing model...")
        best_path = test_trained_model(env, model, start_edge, end_edge)
    
    # Evaluate the best path
    if best_path:
        # Reset environment to evaluate path
        if traci.isConnected():
            traci.close()
        
        env.reset(options={'start_edge': best_path[0], 'end_edge': best_path[-1]})
        metrics = evaluate_path(env, best_path)
        
        print("\n===== Optimal Path Results =====")
        print(f"Path: {' -> '.join(best_path)}")
        print(f"Travel Time: {metrics['travel_time']:.2f} seconds")
        print(f"Total Distance: {metrics['total_distance']:.2f} meters")
        print(f"Traffic Lights: {metrics['traffic_light_count']}")
        print(f"Average Vehicles per Edge: {metrics['avg_vehicles_per_edge']:.2f}")
        print("===============================\n")
        
        # Visualize path
        if visualize:
            visualize_path(net_file, best_path)
            print("Path visualization saved as 'optimal_path.png'")
    else:
        print("Could not find a valid path.")
    
    # Close environment
    env.close()
    
    return best_path, model

def test_trained_model(env, model, start_edge=None, end_edge=None, num_tests=5):
    """Test the trained model to find optimal path"""
    best_path = None
    best_reward = float('-inf')
    
    for _ in range(num_tests):
        # Reset environment
        if start_edge and end_edge:
            obs, _ = env.reset(options={'start_edge': start_edge, 'end_edge': end_edge})
        else:
            obs, _ = env.reset()
        
        # Unwrap observation for CNN
        feature_extractor = CNNFeatureExtractor(env)
        obs = feature_extractor.extract_features(obs)
        
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            
            # Unwrap observation for CNN
            if not done:
                obs = feature_extractor.extract_features(obs)
        
        # Check if path reaches goal
        if info['reached_goal'] and total_reward > best_reward:
            best_reward = total_reward
            best_path = info['path']
    
    return best_path

if __name__ == "__main__":
    # Hardcoded file paths
    sumocfg_file = "C:/Users/Administrator/Desktop/Prof.Mangini/Project2025/b.sumocfg"
    net_file = "C:/Users/Administrator/Desktop/Prof.Mangini/Project2025/b.net.xml"
    route_file = "C:/Users/Administrator/Desktop/Prof.Mangini/Project2025/b.rou.xml"
    
    # Optional parameters
    start_edge = None  # Random if None
    end_edge = None    # Random if None
    training_steps = 50000
    visualize = True
    
    print(f"Using SUMO files:")
    print(f"  Config: {sumocfg_file}")
    print(f"  Network: {net_file}")
    print(f"  Routes: {route_file}")
    
    # Find optimal path
    best_path, model = find_optimal_path(
        sumocfg_file,
        net_file,
        route_file,
        start_edge=start_edge,
        end_edge=end_edge,
        training_steps=training_steps,
        visualize=visualize
    )
    
    # Save model
    model.save("optimal_path_model")
    print("Trained model saved as 'optimal_path_model'")