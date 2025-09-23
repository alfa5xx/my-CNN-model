import sys
import os
sys.path.append(os.path.join(os.environ.get("SUMO_HOME"), 'tools'))
import numpy as np
import gym
from gym import spaces
import traci
import sumolib
import tempfile
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback

class SUMOEnvironment(gym.Env):
    """Custom Environment for SUMO traffic simulation."""
    
    def __init__(self, sumocfg_file, start_node, end_node, gui=False):
        super(SUMOEnvironment, self).__init__()
        
        # SUMO configuration
        self.sumocfg = sumocfg_file
        self.start_node = start_node
        self.end_node = end_node
        self.gui = gui
        self.sumo_cmd = ["sumo-gui" if gui else "sumo", "-c", sumocfg_file]
        
        # Load SUMO network
        self.net = sumolib.net.readNet(os.path.dirname(sumocfg_file) + "C:/Users/Administrator/Desktop/Prof.Mangini/Project2025/b.net.xml")
        
        # Get all edges from the network
        self.edges = self.net.getEdges()
        self.edge_ids = [edge.getID() for edge in self.edges]
        
        # Action space: choose which outgoing edge to take
        # Each action represents selecting one of the possible next edges
        max_out_edges = max(len(edge.getOutgoing()) for edge in self.edges)
        self.action_space = spaces.Discrete(max_out_edges)
        
        # Observation space: current edge, distance to destination, traffic density
        # We'll use a simplified representation
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]), 
            high=np.array([len(self.edges), 10000, 1]), 
            dtype=np.float32
        )
        
        self.current_edge = None
        self.path = []
        self.total_time = 0
        self.total_distance = 0
        self.step_count = 0
        self.max_steps = 100  # Limit episode length
        
    def _get_observation(self):
        """
        Returns the current observation.
        Observation consists of:
        - Current edge index
        - Euclidean distance to destination
        - Current edge traffic density (0-1)
        """
        # Find current edge index
        current_edge_idx = self.edge_ids.index(self.current_edge.getID())
        
        # Calculate Euclidean distance to destination
        dest_node = self.net.getNode(self.end_node)
        dest_x, dest_y = dest_node.getCoord()
        
        curr_edge_shape = self.current_edge.getShape()
        curr_x, curr_y = curr_edge_shape[-1]  # Use the end point of current edge
        
        distance = np.sqrt((curr_x - dest_x)**2 + (curr_y - dest_y)**2)
        
        # Get traffic density (simplified)
        vehicle_count = len(traci.edge.getLastStepVehicleIDs(self.current_edge.getID()))
        max_vehicles = max(1, self.current_edge.getLength() / 5)  # Assume 5m per vehicle
        density = min(1.0, vehicle_count / max_vehicles)
        
        return np.array([current_edge_idx, distance, density], dtype=np.float32)
    
    def reset(self):
        """Reset the environment to start a new episode."""
        # Start SUMO
        if 'SUMO_HOME' not in os.environ:
            os.environ['SUMO_HOME'] = 'C:/Program Files (x86)/Eclipse/Sumo/bin/sumo.exe'  # Update with your SUMO path
            
        traci.start(self.sumo_cmd)
        
        # Initialize state
        start_node = self.net.getNode(self.start_node)
        self.current_edge = start_node.getOutgoing()[0]  # Get first outgoing edge
        
        self.path = [self.current_edge.getID()]
        self.total_time = 0
        self.total_distance = 0
        self.step_count = 0
        
        # Add a vehicle
        veh_id = "rl_vehicle"
        traci.vehicle.add(veh_id, "route_0", departSpeed="0")
        traci.vehicle.setSpeed(veh_id, 13.9)  # ~50 km/h
        
        return self._get_observation()
    
    def step(self, action):
        """Take a step in the environment by choosing an outgoing edge."""
        self.step_count += 1
        
        # Get possible next edges
        outgoing_edges = list(self.current_edge.getOutgoing().values())
        
        # Ensure action is valid
        if action >= len(outgoing_edges):
            action = len(outgoing_edges) - 1
            
        # If there are no outgoing edges, we're stuck
        if not outgoing_edges:
            return self._get_observation(), -100, True, {"message": "No outgoing edges"}
        
        # Select next edge based on action
        next_edge = outgoing_edges[action]
        
        # Get travel time and distance
        edge_id = self.current_edge.getID()
        travel_time = traci.edge.getTraveltime(edge_id)
        travel_dist = self.current_edge.getLength()
        
        # Update current edge
        self.current_edge = next_edge
        self.path.append(next_edge.getID())
        
        # Update metrics
        self.total_time += travel_time
        self.total_distance += travel_dist
        
        # Advance simulation
        traci.simulationStep()
        
        # Check if we've reached the destination
        current_node = self.current_edge.getToNode().getID()
        done = (current_node == self.end_node)
        
        # Also end if we've taken too many steps
        if self.step_count >= self.max_steps:
            done = True
        
        # Compute reward
        if done and current_node == self.end_node:
            # Bonus for reaching the destination
            reward = 100 - 0.1 * self.total_time - 0.01 * self.total_distance
        else:
            # Small penalty for each step to encourage shorter paths
            reward = -1
            
            # Add a directional component to guide towards destination
            dest_node = self.net.getNode(self.end_node)
            dest_x, dest_y = dest_node.getCoord()
            
            current_node = self.current_edge.getToNode()
            curr_x, curr_y = current_node.getCoord()
            
            # Previous node coordinates
            prev_edge = self.net.getEdge(self.path[-2]) if len(self.path) > 1 else None
            if prev_edge:
                prev_node = prev_edge.getFromNode()
                prev_x, prev_y = prev_node.getCoord()
                
                # Check if we're getting closer to destination
                old_dist = np.sqrt((prev_x - dest_x)**2 + (prev_y - dest_y)**2)
                new_dist = np.sqrt((curr_x - dest_x)**2 + (curr_y - dest_y)**2)
                
                if new_dist < old_dist:
                    reward += 1  # Reward for moving towards destination
        
        # If the episode is done, close SUMO
        if done:
            traci.close()
            
        info = {
            "path": self.path,
            "total_time": self.total_time,
            "total_distance": self.total_distance
        }
        
        return self._get_observation(), reward, done, info
    
    def close(self):
        """Close the environment."""
        try:
            traci.close()
        except:
            pass

def create_temp_sumo_files():
    """
    Create temporary SUMO configuration files for testing.
    In a real scenario, you would use your own network files.
    """
    temp_dir = tempfile.mkdtemp()
    
def use_existing_sumo_files():
    base_dir = r"C:/Users/Administrator/Desktop/Prof.Mangini/Project2025"
    sumocfg_file = os.path.join(base_dir, "b.sumocfg")
    
    if not os.path.exists(sumocfg_file):
        # Create the SUMO configuration file if it doesn't exist
        with open(sumocfg_file, 'w') as f:
            f.write("""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="b.net.xml"/>
        <route-files value="b.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="1000"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
    </processing>
    <report>
        <verbose value="false"/>
        <no-step-log value="true"/>
        <no-warnings value="true"/>
    </report>
</configuration>
""")
    
    return sumocfg_file
    
 

def train_rl_model(sumocfg_file, start_node="j0", end_node="j5", total_timesteps=15000):
    """
    Train a RL model for finding the shortest path in SUMO.
    
    Args:
        sumocfg_file: SUMO configuration file
        start_node: Starting node in the network
        end_node: Target node in the network
        total_timesteps: Number of training iterations (default: 15000)
        
    Returns:
        Trained model
    """
    # Create and wrap the environment
    def make_env():
        return SUMOEnvironment(sumocfg_file, start_node, end_node, gui=False)
    
    env = DummyVecEnv([make_env])
    
    # Create the RL model (using PPO algorithm)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        tensorboard_log="./sumo_ppo_tensorboard/"
    )
    
    # Setup checkpointing
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path="./sumo_checkpoints/",
        name_prefix="ppo_sumo_model",
        save_replay_buffer=True,
        save_vecnormalize=True
    )
    
    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    
    # Save the final model
    model.save("sumo_ppo_model_final")
    
    return model

def evaluate_model(model, sumocfg_file, start_node="j0", end_node="j5", episodes=10, gui=True):
    """
    Evaluate the trained model.
    
    Args:
        model: Trained RL model
        sumocfg_file: SUMO configuration file
        start_node: Starting node in the network
        end_node: Target node in the network
        episodes: Number of evaluation episodes
        gui: Whether to show the SUMO GUI during evaluation
    """
    eval_env = SUMOEnvironment(sumocfg_file, start_node, end_node, gui=gui)
    
    # Results tracking
    paths = []
    times = []
    distances = []
    
    for _ in range(episodes):
        obs = eval_env.reset()
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            
            if done:
                paths.append(info["path"])
                times.append(info["total_time"])
                distances.append(info["total_distance"])
    
    eval_env.close()
    
    # Print results
    print(f"Evaluation over {episodes} episodes:")
    print(f"Average travel time: {np.mean(times):.2f} seconds")
    print(f"Average travel distance: {np.mean(distances):.2f} meters")
    
    # Show the best path
    best_idx = np.argmin(times)
    print(f"Best path (travel time: {times[best_idx]:.2f}s, distance: {distances[best_idx]:.2f}m):")
    print(" -> ".join(paths[best_idx]))
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(episodes), times)
    plt.xlabel("Episode")
    plt.ylabel("Travel Time (s)")
    plt.title("Travel Times by Episode")
    
    plt.subplot(1, 2, 2)
    plt.bar(range(episodes), distances)
    plt.xlabel("Episode")
    plt.ylabel("Travel Distance (m)")
    plt.title("Travel Distances by Episode")
    
    plt.tight_layout()
    plt.savefig("sumo_rl_results.png")
    plt.show()

def main():
    """Main function to create environment, train and evaluate model."""
    # Create temporary SUMO files (for testing)
    # In a real scenario, you would use your own SUMO network files
    sumocfg_file = create_temp_sumo_files()
    
def create_temp_sumo_files():
    """
    Create temporary SUMO configuration files for testing.
    In a real scenario, you would use your own network files.
    """
    # Define the base directory where files should be saved
    base_dir = r"C:/Users/Administrator/Desktop/Prof.Mangini/Project2025"
    
    # Make sure the directory exists
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create proper paths for the files
    net_file = os.path.join(base_dir, "b.net.xml")
    routes_file = os.path.join(base_dir, "b.rou.xml")
    sumocfg_file = os.path.join(base_dir, "b.sumocfg")
    
    # Create a simple network (or you can use an existing one)
    with open(net_file, 'w') as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<net xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd" version="1.9.2">
    <!-- Your network XML content here -->
    <!-- Using a minimal example for testing -->
    <location netOffset="0.00,0.00" convBoundary="0.00,0.00,1000.00,1000.00" origBoundary="-10000000000.00,-10000000000.00,10000000000.00,10000000000.00" projParameter="!"/>
    <edge id="e0" from="j0" to="j1" priority="1" length="200.00">
        <lane id="e0_0" index="0" speed="13.89" length="200.00" shape="0.00,0.00 200.00,0.00"/>
    </edge>
    <!-- Add more edges as needed -->
</net>
""")
    
    # Create a routes file
    with open(routes_file, 'w') as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="50" color="1,0,0"/>
    <route id="route_0" edges="e0"/>
</routes>
""")
    
    # Create a SUMO configuration file
    with open(sumocfg_file, 'w') as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="b.net.xml"/>
        <route-files value="b.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="1000"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
    </processing>
    <report>
        <verbose value="false"/>
        <no-step-log value="true"/>
        <no-warnings value="true"/>
    </report>
</configuration>
""")
    
    return sumocfg_file
    
    print("Training RL model for SUMO shortest path finding...")
    model = train_rl_model(sumocfg_file, start_node="j0", end_node="j5", total_timesteps=15000)
    
    print("Evaluating the trained model...")
    evaluate_model(model, sumocfg_file, episodes=5)
    
    print("Done! Model saved as 'sumo_ppo_model_final'")

if __name__ == "__main__":
    main()