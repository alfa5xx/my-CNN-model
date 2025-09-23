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
from typing import Tuple, Dict, Any, List


class SUMOEnv(gym.Env):
    def __init__(self, sumo_config_path: str, gui: bool = False):
        super(SUMOEnv, self).__init__()

        # Store configuration path
        self.sumo_config_path = sumo_config_path
        self.network_path = os.path.join(os.path.dirname(sumo_config_path), "C:/Users/Administrator/Desktop/Prof.Mangini/Project2025/b.net.xml")
        self.gui = gui
        
        # Start SUMO simulation
        self._start_simulation()
        
        # Get network information after SUMO is started
        self.net = sumolib.net.readNet(self.network_path)
        self.edge_list = self._get_all_edges()
        self.traffic_light_ids = traci.trafficlight.getIDList()
        
        # Define action and observation space
        # Actions: Choose from available outgoing edges at each junction
        self.action_space = spaces.Discrete(len(self.edge_list))
        
        # Observation space: Features include:
        # - Current edge occupancy (normalized)
        # - Next 3 potential edges occupancy
        # - Traffic light states for upcoming junctions
        # - Distance to destination
        # - Current speed
        feature_count = 7  # Base features
        feature_count += len(self.traffic_light_ids)  # Add traffic light states
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(feature_count,), dtype=np.float32
        )
        
        # # Environment state tracking original
        # self.vehicle_id = None
        # self.start_edge= "E10"
        # self.destination_edge = "E13"
        # self.current_route = None
        # self.steps_taken = 0
        # self.max_steps = 1000  # Maximum steps before termination
        # self.arrived = False
        # self.traci_running = True
        
                # Environment state tracking
        self.vehicle_id = None
        self.start_edge = "E10"        # Fixed start edge
        self.destination_edge = "E13"    # Fixed destination edge
        self.current_route = None
        self.steps_taken = 0
        self.max_steps = 1000  # Maximum steps before termination
        self.arrived = False
        self.traci_running = True

        

    def _start_simulation(self) -> None:
        """Start the SUMO simulation"""
        if not self.gui:
            sumo_binary = "sumo"
        else:
            sumo_binary = "sumo-gui"
            
        sumo_cmd = [sumo_binary, "-c", self.sumo_config_path, "--no-warnings", "--start"]
        
        # Try to safely close any existing SUMO connections
        try:
            traci.close()
        except:
            pass
            
        traci.start(sumo_cmd)
        self.traci_running = True

    def _get_all_edges(self) -> List[str]:
        """Retrieve all possible edges from SUMO network"""
        return [edge for edge in traci.edge.getIDList()]

    def _get_available_next_edges(self, current_edge: str) -> List[str]:
        """Get possible next edges from current edge"""
        try:
            current_edge_obj = self.net.getEdge(current_edge)
            next_edges = []
            
            # Get all outgoing connections
            for connection in current_edge_obj.getOutgoing():
                next_edge = connection.getTo().getID()
                if next_edge not in next_edges:
                    next_edges.append(next_edge)
                    
            return next_edges
        except:
            # Return all edges if there's an error or no outgoing edges
            return self.edge_list
        ################################################################### new ######
    def _get_optimal_route(self, start_edge: str, destination_edge: str) -> List[str]:
        """Calculate the optimal (shortest) route between the given start and destination edges."""
        try:
            start_edge_obj = self.net.getEdge(start_edge)
            destination_edge_obj = self.net.getEdge(destination_edge)
            path, _ = self.net.getShortestPath(start_edge_obj, destination_edge_obj)
            route_edges = [edge.getID() for edge in path]
            # If no path is found, fallback to directly connecting the edges.
            if not route_edges:
                route_edges = [start_edge, destination_edge]
        except Exception as e:
            print(f"Error calculating optimal route: {e}")
            route_edges = [start_edge, destination_edge]
            return route_edges

  #####################################################
        
def _get_edge_occupancy(self, edge_id: str) -> float:
    """Calculate edge occupancy as vehicles/edge_length"""
    try:
        vehicles = traci.edge.getLastStepVehicleNumber(edge_id)
        edge_length = self.net.getEdge(edge_id).getLength()
        max_possible_vehicles = max(1, edge_length / 5)  # Assuming 5m per vehicle
        return min(1.0, vehicles / max_possible_vehicles)
    except:
        return 0.0


    def _get_traffic_light_states(self) -> List[float]:
        """Get traffic light states normalized to [0,1]"""
        light_states = []
        
        for tl_id in self.traffic_light_ids:
            try:
                # Get current state
                state = traci.trafficlight.getRedYellowGreenState(tl_id)
                
                # Count green lights and normalize by total lights
                green_count = state.count('G') + state.count('g')
                total_lights = len(state)
                
                # Normalize to [0,1]
                normalized_state = green_count / total_lights if total_lights > 0 else 0
                light_states.append(normalized_state)
            except:
                light_states.append(0.0)
                
        return light_states

    def _get_observation(self) -> np.ndarray:
        """Create observation vector based on environment state"""
        observation = []
        
        if not self.traci_running or self.vehicle_id not in traci.vehicle.getIDList():
            # If vehicle doesn't exist, return zeros
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        # Get current edge
        try:
            current_edge = traci.vehicle.getRoadID(self.vehicle_id)
            if current_edge.startswith(':'):  # Junction ID starts with ':'
                # If we're at a junction, use the last edge
                current_edge = self.current_route[-1] if self.current_route else self.edge_list[0]
        except:
            current_edge = self.edge_list[0]
        
        # Current edge occupancy (normalized)
        current_occupancy = self._get_edge_occupancy(current_edge)
        observation.append(current_occupancy)
        
        # Next potential edges occupancy
        next_edges = self._get_available_next_edges(current_edge)
        for i in range(3):  # Consider next 3 potential edges
            if i < len(next_edges):
                observation.append(self._get_edge_occupancy(next_edges[i]))
            else:
                observation.append(0.0)  # Pad with zeros if fewer than 3 next edges
        
        # Get traffic light states
        observation.extend(self._get_traffic_light_states())
        
        # Calculate distance to destination (normalized)
        try:
            current_position = traci.vehicle.getPosition(self.vehicle_id)
            destination_edge_obj = self.net.getEdge(self.destination_edge)
            destination_position = destination_edge_obj.getFromNode().getCoord()
            
            # Calculate Euclidean distance
            distance = np.sqrt((current_position[0] - destination_position[0])**2 + 
                              (current_position[1] - destination_position[1])**2)
            
            # Normalize by maximum possible distance in network
            max_distance = 10000  # Adjust based on your network size
            normalized_distance = min(1.0, distance / max_distance)
            
            observation.append(normalized_distance)
        except:
            observation.append(1.0)  # Maximum normalized distance
        
        # Current speed (normalized)
        try:
            speed = traci.vehicle.getSpeed(self.vehicle_id)
            max_speed = traci.vehicle.getAllowedSpeed(self.vehicle_id)
            normalized_speed = speed / max_speed if max_speed > 0 else 0
            observation.append(normalized_speed)
        except:
            observation.append(0.0)
        
        # Make sure observation matches expected shape
        while len(observation) < self.observation_space.shape[0]:
            observation.append(0.0)
            
        return np.array(observation, dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Apply action and return new state"""
        if not self.traci_running:
            return self._get_observation(), -10.0, True, False, {"message": "TRACI not running"}
            
        self.steps_taken += 1
        
        # Check if vehicle exists
        if self.vehicle_id not in traci.vehicle.getIDList():
            print(f"Warning: Vehicle {self.vehicle_id} not found in simulation!")
            return self._get_observation(), -10.0, True, False, {"message": "Vehicle not found"}
        
        # Get current edge
        try:
            current_edge = traci.vehicle.getRoadID(self.vehicle_id)
            if current_edge.startswith(':'):  # Junction ID starts with ':'
                # If we're at a junction, use the last edge
                current_edge = self.current_route[-1] if self.current_route else self.edge_list[0]
                
            # Update current route
            if self.current_route is None:
                self.current_route = [current_edge]
            elif self.current_route[-1] != current_edge and not current_edge.startswith(':'):
                self.current_route.append(current_edge)
        except:
            print("Error getting current edge")
        
        # Get available next edges
        next_edges = self._get_available_next_edges(current_edge)
        
        if next_edges:
            # Ensure action is valid
            target_edge = self.edge_list[action % len(self.edge_list)]
            
            # Check if target edge is a valid next edge
            if target_edge not in next_edges and len(next_edges) > 0:
                target_edge = next_edges[action % len(next_edges)]
                
            try:
                # Change vehicle's target edge
                traci.vehicle.changeTarget(self.vehicle_id, target_edge)
            except:
                print(f"Error changing target to {target_edge}")
        
        # Execute simulation step
        traci.simulationStep()
        
        # Check if vehicle has arrived
        arrived_list = traci.simulation.getArrivedIDList()
        self.arrived = self.vehicle_id in arrived_list
        
        # Check termination conditions
        done = (self.arrived or 
                self.steps_taken >= self.max_steps or 
                self.vehicle_id not in traci.vehicle.getIDList())
        
        # Calculate reward
        reward = self._calculate_reward()
        
        return self._get_observation(), reward, done, False, {"arrived": self.arrived}

    def _calculate_reward(self) -> float:
        """Calculate reward based on traffic, progress, and objective"""
        reward = 0.0
        
        if not self.traci_running:
            return -10.0
            
        # Check if vehicle exists
        if self.vehicle_id not in traci.vehicle.getIDList():
            return -10.0  # Penalize for vehicle disappearing
        
        try:
            # Reward for being on low-traffic edges
            current_edge = traci.vehicle.getRoadID(self.vehicle_id)
            if not current_edge.startswith(':'):  # Not a junction
                occupancy = self._get_edge_occupancy(current_edge)
                reward -= occupancy * 2  # Higher penalty for congested roads
            
            # Reward for speed (normalized by max allowed speed)
            speed = traci.vehicle.getSpeed(self.vehicle_id)
            max_speed = traci.vehicle.getAllowedSpeed(self.vehicle_id)
            speed_reward = speed / max_speed if max_speed > 0 else 0
            reward += speed_reward
            
            # Reward for progress toward destination
            distance_to_dest = traci.simulation.getDistance2D(
                traci.vehicle.getPosition(self.vehicle_id)[0],
                traci.vehicle.getPosition(self.vehicle_id)[1],
                self.net.getEdge(self.destination_edge).getFromNode().getCoord()[0],
                self.net.getEdge(self.destination_edge).getFromNode().getCoord()[1],
                False  # Aerial distance
            )
            
            # Store initial distance for comparison
            if not hasattr(self, 'initial_distance'):
                self.initial_distance = distance_to_dest
                
            # Calculate progress as a percentage of initial distance
            if self.initial_distance > 0:
                progress = (self.initial_distance - distance_to_dest) / self.initial_distance
                reward += progress
            
            # Large reward for reaching destination
            if self.arrived:
                reward += 100.0
                
            # Small penalty for each step to encourage shorter routes
            reward -= 0.1
            
        except:
            # Fail-safe for errors
            reward -= 1.0
            
        return reward
################################# old --- 

    # def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
    #     """Reset environment and generate a random route"""
    #     # Reset environment state
    #     self.steps_taken = 0
    #     self.arrived = False
    #     self.current_route = None
        
    #     # Close and restart SUMO
    #     try:
    #         traci.close()
    #         self.traci_running = False
    #     except:
    #         pass
            
    #     self._start_simulation()
        
    #     # Generate a random vehicle ID
    #     self.vehicle_id = self._generate_vehicle_id()
        
    #     # Load network
    #     self.net = sumolib.net.readNet(self.network_path)
        
    #     # Generate a random route
    #     route_edges = self._get_random_route()
    #     route_id = f"route_{self.vehicle_id}"
        
    #     # Set destination edge
    #     self.destination_edge = route_edges[-1]
        
    #     # Add route to SUMO
    #     try:
    #         traci.route.add(route_id, route_edges)
    #     except:
    #         print("Error adding route, trying again with different edges")
    #         route_edges = [random.choice(self.edge_list), random.choice(self.edge_list)]
    #         traci.route.add(route_id, route_edges)
    #         self.destination_edge = route_edges[-1]
            
    #     # Add vehicle with the random route
    #     try:
    #         traci.vehicle.add(
    #             self.vehicle_id, 
    #             route_id,
    #             departSpeed="max",
    #             departLane="best"
    #         )
    #     except:
    #         print(f"Error adding vehicle {self.vehicle_id}")
        
    #     # Advance simulation slightly to place vehicle
    #     for _ in range(5):
    #         traci.simulationStep()
            
    #     return self._get_observation(), {}
    #######################################################
    
    ############################# second edit ############################################
    
    # def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
    #     """Reset environment and generate the optimal route based on start and destination edges."""
    # # Reset environment state
    # self.steps_taken = 0
    # self.arrived = False
    # self.current_route = None

    # # Close and restart SUMO
    # try:
    #     traci.close()
    #     self.traci_running = False
    # except:
    #     pass

    # self._start_simulation()

    # # Generate a random vehicle ID
    # self.vehicle_id = self._generate_vehicle_id()

    # # Load network
    # self.net = sumolib.net.readNet(self.network_path)

    # # Generate the optimal route using fixed start and destination edges
    # route_edges = self._get_optimal_route(self.start_edge, self.destination_edge)
    # route_id = f"route_{self.vehicle_id}"

    # # Do not override the fixed destination_edge here
    # try:
    #     traci.route.add(route_id, route_edges)
    # except Exception as e:
    #     print("Error adding route, trying fallback with direct connection:", e)
    #     route_edges = [self.start_edge, self.destination_edge]
    #     traci.route.add(route_id, route_edges)

    # # Add vehicle with the optimal route
    # try:
    #     traci.vehicle.add(
    #         self.vehicle_id, 
    #         route_id,
    #         departSpeed="max",
    #         departLane="best"
    #     )
    # except:
    #     print(f"Error adding vehicle {self.vehicle_id}")

    # # Advance simulation slightly to place vehicle
    # for _ in range(5):
    #     traci.simulationStep()

    # return self._get_observation(), {}

#######################################################  third  in below ####
def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Reset environment and generate a random route (or optimal route if you prefer)"""
    # Reset environment state
    self.steps_taken = 0
    self.arrived = False
    self.current_route = None

    # Safely close any existing connection
    try:
        traci.close()
        self.traci_running = False
    except Exception as e:
        print("Error closing traci:", e)

    # Restart SUMO simulation
    self._start_simulation()

    # Generate a random vehicle ID
    self.vehicle_id = self._generate_vehicle_id()

    # Load network again (in case it changed)
    self.net = sumolib.net.readNet(self.network_path)

    # Generate a random route (or optimal route based on start/destination)
    try:
        route_edges = self._get_random_route()
    except Exception as e:
        print("Error generating route:", e)
        # Fallback to a simple route (ensure there are at least 2 edges in self.edge_list)
        route_edges = [random.choice(self.edge_list), random.choice(self.edge_list)]
    
    route_id = f"route_{self.vehicle_id}"
    # Set destination edge based on the route
    self.destination_edge = route_edges[-1]

    # Add the route to SUMO
    try:
        traci.route.add(route_id, route_edges)
    except Exception as e:
        print("Error adding route, using fallback route:", e)
        # Fallback route: choose two random edges
        route_edges = [random.choice(self.edge_list), random.choice(self.edge_list)]
        traci.route.add(route_id, route_edges)
        self.destination_edge = route_edges[-1]

    # Add the vehicle with the route
    try:
        traci.vehicle.add(self.vehicle_id, route_id, departSpeed="max", departLane="best")
    except Exception as e:
        print(f"Error adding vehicle {self.vehicle_id}:", e)

    # Advance simulation a few steps to place the vehicle
    for _ in range(5):
        traci.simulationStep()

    # Get observation; if None, fallback to a zero vector matching the observation space
    observation = self._get_observation()
    if observation is None:
        observation = np.zeros(self.observation_space.shape[0], dtype=np.float32)

    # Always return a tuple (observation, info)
    return observation, {}

############################################



    
    
    

    def _generate_vehicle_id(self) -> str:
        """Generate a random 5-character vehicle ID"""
        return 'veh_' + ''.join(random.choices(string.ascii_letters + string.digits, k=5))

    def _get_random_route(self) -> List[str]:
        """Generate a random route based on SUMO network"""
        edges = list(self.net.getEdges())
        
        if len(edges) < 2:
            raise ValueError("Network must have at least two edges.")
            
        # Randomly select start and end edges (ensuring they are different)
        from_edge = random.choice(edges)
        to_edge = random.choice(edges)
        
        while to_edge == from_edge:
            to_edge = random.choice(edges)
            
        # Find shortest path between selected edges
        try:
            path, _ = self.net.getShortestPath(from_edge, to_edge)
            route_edges = [edge.getID() for edge in path]
            
            # Ensure route has at least start and end
            if not route_edges:
                route_edges = [from_edge.getID(), to_edge.getID()]
        except:
            # Fallback for path finding errors
            route_edges = [from_edge.getID(), to_edge.getID()]
            
        return route_edges

    def close(self) -> None:
        """Close SUMO environment"""
        try:
            traci.close()
            self.traci_running = False
        except:
            pass


def train_agent(env_config_path: str, output_model_path: str, total_timesteps: int = 100000, gui: bool = False):
    """Train a PPO agent for optimal path finding in SUMO"""
    # Create environment
    env = SUMOEnv(env_config_path, gui=gui)
    
    # Create and train PPO agent
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        tensorboard_log="./ppo_sumo_tensorboard/"
    )
    
    print(f"Starting training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    
    # Save the trained model
    model.save(output_model_path)
    print(f"Model saved to {output_model_path}")
    
    # Close environment
    env.close()
    
    return model


def evaluate_agent(model_path: str, env_config_path: str, episodes: int = 10, gui: bool = True):
    """Evaluate a trained agent on the SUMO environment"""
    # Load model
    model = PPO.load(model_path)
    
    # Create environment with GUI
    env = SUMOEnv(env_config_path, gui=gui)
    
    # Evaluation metrics
    total_rewards = []
    success_rate = 0
    
    for episode in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Use model to predict action
            action, _ = model.predict(obs, deterministic=True)
            
            # Apply action
            obs, reward, done, _, info = env.step(action)
            
            episode_reward += reward
            
            # Check if destination reached
            if info.get("arrived", False):
                success_rate += 1
                
        total_rewards.append(episode_reward)
        print(f"Episode {episode+1}/{episodes}: Reward = {episode_reward}, Arrived = {info.get('arrived', False)}")
    
    # Calculate statistics
    avg_reward = sum(total_rewards) / episodes
    success_percentage = (success_rate / episodes) * 100
    
    print(f"\nEvaluation Results:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Success Rate: {success_percentage:.2f}%")
    
    # Close environment
    env.close()


if __name__ == "__main__":
    # Update these paths to match your SUMO configuration
    sumo_config_path = "C:/Users/Administrator/Desktop/Prof.Mangini/Project2025/b.sumocfg"
    model_output_path = "ppo_sumo_optimal_path"
    
    # Train the agent (set gui=True to visualize training)
    trained_model = train_agent(
        env_config_path=sumo_config_path,
        output_model_path=model_output_path,
        total_timesteps=100000,  # Increase for better performance
        gui=False  # Set to True for visualization during training
    )
    
    # Evaluate the trained agent
    evaluate_agent(
        model_path=model_output_path,
        env_config_path=sumo_config_path,
        episodes=10,
        gui=True  # Set to True to visualize evaluation
    )