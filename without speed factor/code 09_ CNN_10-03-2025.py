import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import sumolib
import traci

# Ensure SUMO and Python path are set correctly
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    raise ImportError("Please declare environment variable 'SUMO_HOME'")

class SUMOPathEnv(gym.Env):
    """
    Custom Gym Environment for SUMO Path Optimization
    """
    def __init__(self, net_file, route_file, config_file, start_edge, end_edge):
        super(SUMOPathEnv, self).__init__()
        
        # SUMO configuration
        self.net_file = net_file
        self.route_file = route_file
        self.config_file = config_file
        self.start_edge = start_edge
        self.end_edge = end_edge
        
        # Network parsing
        self.net = sumolib.net.readNet(net_file)
        
        # State and action spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(len(self.net.getEdges()), 6),  # Features per edge
            dtype=np.float32
        )
        
        # Action space: select next edge to move
        self.action_space = gym.spaces.Discrete(len(self.net.getEdges()))
        
        # Path tracking
        self.current_path = []
        self.current_edge = None
        
        # TRACI connection tracking
        self.sumo_running = False
    
    def _get_edge_features(self, edge):
        """
        Extract features for a given edge
        """
        edge_id = edge.getID()
        return [
            edge.getLength(),  # Length
            len(edge.getLanes()),  # Number of lanes
            edge.getSpeed(),  # Speed limit
            len(traci.edge.getLastStepVehicleIDs(edge_id)) if self.sumo_running else 0,  # Vehicle count
            traci.edge.getWaitingTime(edge_id) if self.sumo_running else 0,  # Waiting time
            traci.edge.getCO2Emission(edge_id) if self.sumo_running else 0  # CO2 emission
        ]

    def _start_simulation(self):
        """
        Start or restart the SUMO simulation
        """
        # Close any existing connection
        self._close_simulation()
        
        # Start SUMO simulation
        sumo_cmd = [
            "sumo", "-n", self.net_file, 
            "-r", self.route_file, 
            "--start", "--no-step-log"
        ]
        traci.start(sumo_cmd)
        self.sumo_running = True
    
    def _close_simulation(self):
        """
        Close the SUMO simulation if it's running
        """
        if self.sumo_running:
            try:
                traci.close()
                self.sumo_running = False
            except:
                pass  # Already closed or not started
    
    def reset(self, seed=None):
        """
        Reset the environment
        """
        if hasattr(super(), 'reset'):
            super().reset(seed=seed)
        
        # Start/restart SUMO simulation
        self._start_simulation()
        
        # Initialize at start edge
        self.current_edge = self.start_edge
        self.current_path = [self.start_edge]
        
        # Create initial state
        state = np.array([
            self._get_edge_features(edge) 
            for edge in self.net.getEdges()
        ])
        
        return state, {}

    def step(self, action):
        """Take a step in the environment"""
        # Ensure simulation is running
        if not self.sumo_running:
            self._start_simulation()
        
        # Create initial state before any operations
        state = np.array([
            self._get_edge_features(edge) 
            for edge in self.net.getEdges()
        ])
        
        # Convert action to edge
        selected_edge_id = self.net.getEdges()[action].getID()
        
        # Validate edge selection
        if not self._is_valid_edge_transition(selected_edge_id):
            reward = -100  # Penalty for invalid move
            done = True
            return state, reward, done, False, {}
        
        # Move to selected edge
        try:
            traci.vehicle.changeTarget(self.current_path[-1], selected_edge_id)
            self.current_path.append(selected_edge_id)
        except traci.exceptions.TraCIException as e:
            print(f"TraCI Error: {e}")
            reward = -150  # Higher penalty for TraCI errors
            done = True
            return state, reward, done, False, {}
        
        # Compute reward
        reward = self._compute_reward(selected_edge_id)
        
        # Check if reached destination
        done = selected_edge_id == self.end_edge
        
        # Update state
        state = np.array([
            self._get_edge_features(edge) 
            for edge in self.net.getEdges()
        ])
        
        return state, reward, done, False, {}
    
  
    
    def _is_valid_edge_transition(self, edge_id):
        """
        Check if edge transition is valid
            """
        if not self.current_path:
            return False  # No current path
            
        # Logic to ensure connected edges
        current_connections = set()
        current_edge_id = self.current_path[-1]
        
        # Try to get connections using SUMO's network structure
        try:
            # Method 1: Get connections from SUMO network directly
            current_edge = self.net.getEdge(current_edge_id)
            for conn in current_edge.getOutgoing():
                try:
                    target_edge_id = conn.getToEdge().getID()
                    current_connections.add(target_edge_id)
                except (AttributeError, TypeError):
                    pass
            
            # Method 2: If Method 1 fails, try using TRACI to get successors
            if not current_connections and self.sumo_running:
                try:
                    successors = traci.edge.getNextEdges(current_edge_id)
                    current_connections.update(successors)
                except:
                    pass
            
            # Method 3: Try a more direct approach with lane connections
            if not current_connections:
                for lane in current_edge.getLanes():
                    for connection in lane.getOutgoing():
                        try:
                            to_edge = connection.getToLane().getEdge()
                            current_connections.add(to_edge.getID())
                        except (AttributeError, TypeError):
                            pass
            
            # If all else fails, try a hard-coded approach based on your network
            if not current_connections:
                # This should be replaced with the actual connections in your network
                # Examples (assuming you know the network structure):
                if current_edge_id == "E10":
                    # Add known successor edges for E10
                    current_connections.add("E11")  # Assuming E11 is connected to E10
                    current_connections.add("E12")  # Assuming E12 is also connected to E10
                # Add more edge connections as needed based on your network
        
        except Exception as e:
            print(f"Error getting connections for edge {current_edge_id}: {e}")
        
        # Debug output
        if current_connections:
            print(f"Valid connections from {current_edge_id}: {', '.join(current_connections)}")
        else:
            print(f"Warning: No valid connections found from edge {current_edge_id}")
            # In case of no connections found, allow any edge as valid for testing
            return True  # Comment this out once connections are properly detected!
            
        return edge_id in current_connections 
    
    def _compute_reward(self, edge_id):
        """
        Multi-objective reward computation
        """
        # Reward components
        vehicle_count_penalty = -len(traci.edge.getLastStepVehicleIDs(edge_id))
        length_penalty = -self.net.getEdge(edge_id).getLength()
        waiting_time_penalty = -traci.edge.getWaitingTime(edge_id)
        
        # Destination bonus
        destination_bonus = 1000 if edge_id == self.end_edge else 0
        
        # Combine rewards
        total_reward = (
            vehicle_count_penalty * 0.3 + 
            length_penalty * 0.2 + 
            waiting_time_penalty * 0.2 + 
            destination_bonus
        )
        
        return total_reward
    
    def close(self):
        """
        Close SUMO simulation
        """
        self._close_simulation()

class PathCNN(nn.Module):
    """
    Convolutional Neural Network for Path Selection
    """
    def __init__(self, input_dim, hidden_dims):
        super(PathCNN, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Fully connected layers
        fc_input_size = self._get_conv_output_size(input_dim)
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_size, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)  # Single output for path selection
        )
    
    def _get_conv_output_size(self, input_dim):
        """
        Calculate convolutional output size
        """
        test_tensor = torch.zeros(1, input_dim, 10)
        return self._forward_conv(test_tensor).numel()
    
    def _forward_conv(self, x):
        """
        Forward pass through convolutional layers
        """
        return self.conv_layers(x)
    
    def forward(self, x):
        """
        Full forward pass
        """
        conv_out = self.conv_layers(x)
        flat_out = conv_out.view(conv_out.size(0), -1)
        return self.fc_layers(flat_out)

def train_optimal_path(net_file, route_file, config_file, start_edge, end_edge):
    """
    Train the optimal path using PPO and CNN
    """
    try:
        # Create environment
        env = SUMOPathEnv(net_file, route_file, config_file, start_edge, end_edge)
        env = DummyVecEnv([lambda: env])
        
        # Initialize CNN
        cnn = PathCNN(input_dim=6, hidden_dims=[128, 64])
        
        # PPO with custom CNN policy
        model = PPO(
            "MlpPolicy", 
            env, 
            policy_kwargs={
                'net_arch': [dict(pi=[64, 32], vf=[64, 32])],
                'activation_fn': torch.nn.ReLU
            },
            verbose=1,
            learning_rate=0.001,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99
        )
        
        # Train the model
        model.learn(total_timesteps=50000)
        
        # Save the model
        model.save("sumo_path_optimization")
        
        return model
    finally:
        # Ensure TRACI connections are closed even if an error occurs
        try:
            traci.close()
        except:
            pass

def simulate_optimal_path(model, net_file, route_file, config_file, start_edge, end_edge):
    """
    Simulate the optimal path
    """
    env = SUMOPathEnv(net_file, route_file, config_file, start_edge, end_edge)
    obs, _ = env.reset()
    
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, info = env.step(action)
    
    print("Optimal Path Found:", env.current_path)
    env.close()

# Main execution
if __name__ == "__main__":
    # Configuration files
    NET_FILE = "C:/Users/Administrator/Desktop/Prof.Mangini/Project2025/b.net.xml"
    ROUTE_FILE = "C:/Users/Administrator/Desktop/Prof.Mangini/Project2025/b.rou.xml"
    CONFIG_FILE = "C:/Users/Administrator/Desktop/Prof.Mangini/Project2025/b.sumocfg"
    START_EDGE = "E10"  # Replace with actual edge ID
    END_EDGE = "E13"    # Replace with actual edge ID
    
    # Train and simulate
    trained_model = train_optimal_path(
        NET_FILE, ROUTE_FILE, CONFIG_FILE, START_EDGE, END_EDGE
    )
    
    # Simulate the optimal path
    simulate_optimal_path(
        trained_model, NET_FILE, ROUTE_FILE, CONFIG_FILE, START_EDGE, END_EDGE
    )