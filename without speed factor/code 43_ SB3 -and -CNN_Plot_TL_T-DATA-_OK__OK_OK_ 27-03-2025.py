# Import necessary libraries for TensorFlow/Keras, Stable Baselines3, Gym, and SUMO simulation
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt

# SUMO libraries
import traci
import sumolib
from collections import defaultdict

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Import gym and SB3 for reinforcement learning
import gym
from gym import spaces
from stable_baselines3 import PPO

# ------------------------------------------------------------------------------
# Configuration parameters
TRAIN_EPOCHS = 150            # Number of epochs for training CNN
TRAIN_SIM_STEPS = 200      # Simulation steps for training data collection
TEST_SIM_STEPS = 40       # Simulation steps for test data collection
MODEL_CHOICE = "original"  # Options: "original" or "lightweight"
SHOW_PLOT = True           # If True, display training plot interactively

# ------------------------------------------------------------------------------
# Define file paths for the SUMO simulation files
net_file = "C:/Users/Studente/Desktop/Codes/Project2025/b.net.xml"      # SUMO network file
route_file = "C:/Users/Studente/Desktop/Codes/Project2025/b.rou.xml"    # Routes file
sumocfg_file = "C:/Users/Studente/Desktop/Codes/Project2025/b.sumocfg"  # Simulation configuration file

# ------------------------------------------------------------------------------
# Retrieve all edge IDs from the SUMO network using sumolib.
net = sumolib.net.readNet(net_file)
edge_list = [edge.getID() for edge in net.getEdges()]
n_edges = len(edge_list)
print(f"Retrieved {n_edges} edge IDs from SUMO map")

# Create a mapping between edge IDs and their indices
edge_to_index = {edge_id: idx for idx, edge_id in enumerate(edge_list)}
index_to_edge = {idx: edge_id for idx, edge_id in enumerate(edge_list)}

# Create a connectivity map of the network
connectivity_map = defaultdict(list)
for edge in net.getEdges():
    edge_id = edge.getID()
    for connection in edge.getOutgoing():
        connectivity_map[edge_id].append(connection.getID())

# Calculate edge lengths and store them
edge_lengths = {edge.getID(): edge.getLength() for edge in net.getEdges()}

# ------------------------------------------------------------------------------
# Define a custom Gym environment for Traffic Light Prediction using SB3
class TrafficLightEnv(gym.Env):
    
   # A custom environment for predicting the number of traffic lights on an edge.
    #Observation: [number_of_lanes, average_lane_length]
   # Action: Discrete value representing the predicted traffic light count (0 to 10)
    
    def __init__(self):
        super(TrafficLightEnv, self).__init__()
        self.observation_space = spaces.Box(low=0, high=1000, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(11)
        
    def reset(self):
        return np.zeros(2, dtype=np.float32)
    
    def step(self, action):
        observation = np.zeros(2, dtype=np.float32)
        reward = 0.0
        done = True
        info = {}
        return observation, reward, done, info

# Initialize the SB3 environment and PPO model for traffic light prediction.
env = TrafficLightEnv()
sb3_tls_model = PPO('MlpPolicy', env, verbose=0)
# To load a pre-trained model, uncomment the following line:
# sb3_tls_model = PPO.load("sb3_tls_model.zip")

# ------------------------------------------------------------------------------
# Function to predict the number of traffic lights on an edge using the SB3 model
def predict_tls_count_sb3(edge_id):
    
    # Predict the number of traffic lights on a given edge using the SB3 model.
    # The observation is composed of the number of lanes and the average lane length.
   
    try:
        edge_obj = net.getEdge(edge_id)
        lanes = edge_obj.getLanes()
        num_lanes = len(lanes)
        if num_lanes == 0:
            return 0
        lane_lengths = [lane.getLength() for lane in lanes]
        avg_lane_length = np.mean(lane_lengths)
        obs = np.array([num_lanes, avg_lane_length], dtype=np.float32).reshape(1, -1)
        action, _ = sb3_tls_model.predict(obs, deterministic=True)
        predicted_tls = int(action)
        return predicted_tls
    except Exception as e:
        print(f"Error in SB3 TLS prediction for edge {edge_id}: {e}")
        return 0

# ------------------------------------------------------------------------------
# Feature extraction function to get real data from SUMO simulation, with SB3 integration
def extract_features(edge_id):
    """
    Extract features for a given edge from the SUMO simulation.
    Returns a 3-channel image (64x64 grid) with:
    - Channel 1: vehicle density (normalized)
    - Channel 2: traffic light count (normalized)
    - Channel 3: speed factor (normalized)
    """
    try:
        grid_size = (64, 64)
        # 1. Vehicle density
        vehicles = traci.edge.getLastStepVehicleNumber(edge_id)
        edge_length = edge_lengths[edge_id]
        vehicle_density = vehicles / max(edge_length, 1)
        
        # 2. Traffic lights count using SB3 model prediction
        predicted_tls = predict_tls_count_sb3(edge_id)
        edge_obj = net.getEdge(edge_id)
        lanes = [lane.getID() for lane in edge_obj.getLanes()]
        tls_value = predicted_tls / max(len(lanes), 1)
        
        # # 3. Speed factor (current speed divided by max speed)
        # current_speed = traci.edge.getLastStepMeanSpeed(edge_id)
        # max_speed = edge_obj.speed
        # speed_factor = current_speed / max_speed if max_speed > 0 else 0
        
        vehicle_grid = np.ones(grid_size) * vehicle_density
        tls_grid = np.ones(grid_size) * tls_value
        # speed_grid = np.ones(grid_size) * speed_factor
        
        return np.stack([vehicle_grid, tls_grid], axis=-1)
        # return np.stack([vehicle_grid, tls_grid, speed_grid], axis=-1) # original 
    except Exception as e:
        print(f"Error extracting features for edge {edge_id}: {e}")
        return np.zeros((64, 64, 3))

# ------------------------------------------------------------------------------
# Path evaluation function combining multiple criteria
def evaluate_path(path):
    """
    Evaluate a path based on:
      - Total distance (edge lengths)
      - Total number of vehicles (waiting time indicator)
      - Total traffic lights count
      - Number of edges (path length)
    Lower score is better.
    """
    try:
        if not path:
            return float('inf')
        total_length = sum(edge_lengths[edge] for edge in path)
        total_vehicles = sum(traci.edge.getLastStepVehicleNumber(edge) for edge in path)
        total_tls = sum(predict_tls_count_sb3(edge) for edge in path)
        # The number of edges directly represents the path length
        num_edges = len(path)
        
        # Weight factors (can be adjusted based on simulation characteristics)
        length_weight = 0.3
        vehicle_weight = 0.3
        tls_weight = 0.2
        edge_count_weight = 0.2
        
        score = (length_weight * total_length +
                 vehicle_weight * total_vehicles +
                 tls_weight * total_tls +
                 edge_count_weight * num_edges)
        return score
    except Exception as e:
        print(f"Error evaluating path: {e}")
        return float('inf')

# ------------------------------------------------------------------------------
# Function to generate all possible loop-free paths (DFS with max_depth limit)
def generate_all_paths(start_edge, destination_edge, max_depth=20):
    """
    Generate all possible loop-free paths from start_edge to destination_edge.
    """
    all_paths = []
    def dfs(current_edge, current_path, visited):
        if len(current_path) > max_depth or current_edge in visited:
            return
        current_path.append(current_edge)
        visited.add(current_edge)
        if current_edge == destination_edge:
            all_paths.append(list(current_path))
        else:
            for next_edge in connectivity_map[current_edge]:
                dfs(next_edge, current_path, visited)
        current_path.pop()
        visited.remove(current_edge)
    dfs(start_edge, [], set())
    print(f"DEBUG: Generated {len(all_paths)} paths from {start_edge} to {destination_edge}")
    return all_paths


#-------------NEW --------- new code for save  all  generated routes during the training process 27-03-2025----------------

# Global variable to record all routes generated during training
TRAINING_ROUTES = []

def record_training_routes(routes):
    
   # Record and save routes generated during the training process.
    
    #Parameters:
     #   routes (list): A list of routes (each route is a list of edge IDs)
      #                 to be recorded.
    
    global TRAINING_ROUTES
    TRAINING_ROUTES.extend(routes)



#------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Data Collection Functions:
def collect_training_data(start_edge, destination_edge, simulation_steps=TRAIN_SIM_STEPS):
    
   # Generate training data by evaluating multiple paths and selecting optimal segments.
    #Returns feature matrices (X_data) and one-hot labels (Y_data) for the next edge.
    
    X_data = []
    Y_data = []
    try:
        traci.start(sumo_cmd)
        print("SUMO started for training data collection")
        all_paths = generate_all_paths(start_edge, destination_edge)
        if not all_paths:
            print("No valid paths found. Check connectivity.")
            traci.close()
            return np.array([]), np.array([])
        for step in range(simulation_steps):
            traci.simulationStep()
            if step % 5 != 0:
                continue
            print(f"Training simulation step {step}/{simulation_steps}")
            path_scores = [(path, evaluate_path(path)) for path in all_paths]
            # Sort paths by evaluation score (lower is better)
            path_scores.sort(key=lambda x: x[1])
            best_paths = [p for p, _ in path_scores[:min(5, len(path_scores))]]

#--------------------------------------------------- for recording all paths-------------------------------

            # Record the best paths generated in this simulation step
            record_training_routes(best_paths)

#-------------------------------------------------------------------------------------------------------------            
            
            for path in best_paths:
                for i in range(len(path) - 1):
                    current_edge = path[i]
                    next_edge = path[i + 1]
                    features = extract_features(current_edge)
                    X_data.append(features)
                    label = np.zeros(n_edges)
                    label[edge_to_index[next_edge]] = 1
                    Y_data.append(label)
        print(f"Collected {len(X_data)} training samples")
        traci.close()
        return np.array(X_data), np.array(Y_data)
    except Exception as e:
        print(f"Error collecting training data: {e}")
        if traci.isLoaded():
            traci.close()
        return np.array([]), np.array([])

#--------------------------------------------------------------------------------------

def collect_test_data(start_edge, destination_edge, simulation_steps=TEST_SIM_STEPS):
    """
    Collect test data similar to training data collection.
    """
    X_test = []
    Y_test = []
    try:
        traci.start(sumo_cmd)
        print("SUMO started for test data collection")
        all_paths = generate_all_paths(start_edge, destination_edge)
        if not all_paths:
            print("No valid paths found for testing.")
            traci.close()
            return np.array([]), np.array([])
        for step in range(simulation_steps):
            traci.simulationStep()
            if step % 2 != 0:
                continue
            print(f"Test simulation step {step}/{simulation_steps}")
            path_scores = [(path, evaluate_path(path)) for path in all_paths]
            path_scores.sort(key=lambda x: x[1])
            best_paths = [p for p, _ in path_scores[:min(3, len(path_scores))]]
            for path in best_paths:
                for i in range(len(path) - 1):
                    current_edge = path[i]
                    next_edge = path[i + 1]
                    features = extract_features(current_edge)
                    X_test.append(features)
                    label = np.zeros(n_edges)
                    label[edge_to_index[next_edge]] = 1
                    Y_test.append(label)
        print(f"Collected {len(X_test)} test samples")
        traci.close()
        return np.array(X_test), np.array(Y_test)
    except Exception as e:
        print(f"Error collecting test data: {e}")
        if traci.isLoaded():
            traci.close()
        return np.array([]), np.array([])

# ------------------------------------------------------------------------------
# Define CNN model architectures with BatchNormalization to help training
def build_original_model(input_shape):
    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_edges, activation='softmax'))
    return model

def build_lightweight_model(input_shape):
    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(n_edges, activation='softmax'))
    return model

# ------------------------------------------------------------------------------
# Train the CNN model with adjustable model choice, epochs, and callbacks
def train_cnn_model(X_train, Y_train, X_test, Y_test):
    if len(X_train) == 0 or len(Y_train) == 0:
        print("No training data available.")
        return None
    
    print(f"Training CNN model with {len(X_train)} samples")
    print(f"Input shape: {X_train.shape}, Output shape: {Y_train.shape}")
    input_shape = X_train.shape[1:]
    
    if MODEL_CHOICE == "lightweight":
        print("Using lightweight CNN model for faster training.")
        model = build_lightweight_model(input_shape)
    else:
        print("Using original CNN model.")
        model = build_original_model(input_shape)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
    
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=TRAIN_EPOCHS,
        batch_size=64,
        callbacks=[early_stop, reduce_lr]
    )
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    if SHOW_PLOT:
        plt.show()
    else:
        plt.close()
    
    return model

# ------------------------------------------------------------------------------
# New function: Find the optimal path from start_edge to destination_edge
def find_optimal_path_all(model, start_edge, destination_edge):
    """
    Generate all possible paths from start_edge to destination_edge (loop-free) and select the optimal path.
    The optimal path minimizes a composite cost:
      - Total distance (edge lengths)
      - Total vehicles on the path
      - Total traffic lights along the path
      - Number of edges in the path
    """
    try:
        traci.start(sumo_cmd)
        print("SUMO started for optimal path search")
        all_paths = generate_all_paths(start_edge, destination_edge, max_depth=50)
        if not all_paths:
            print("No valid paths found from start to destination.")
            traci.close()
            return []
        # Evaluate each path and choose the one with minimum cost
        optimal_path = min(all_paths, key=lambda path: evaluate_path(path))
        traci.close()
        print(f"Optimal path found: {optimal_path}")
        return optimal_path
    except Exception as e:
        print(f"Error finding optimal path: {e}")
        if traci.isLoaded():
            traci.close()
        return []
#----------------------------------- new code for counting the traffic lights -----------------------------    
def count_traffic_lights_in_path(path):
    """
    Count the total number of traffic lights along a given path.
    
    For each edge in the path, the function checks whether the destination node (junction)
    is controlled by a traffic light. It uses the node type from the SUMO network, and if 
    the simulation is running (via traci), it also verifies using traci.trafficlight.getIDList().
    
    Parameters:
        path (list): A list of edge IDs representing the route.
    
    Returns:
        int: Total count of traffic lights in the path.
    """
    tls_count = 0
    for edge_id in path:
        try:
            edge_obj = net.getEdge(edge_id)
            to_node = edge_obj.getToNode()
            # Check if the destination node is a traffic light junction
            if to_node.getType() == "traffic_light" or (traci.isLoaded() and to_node.getID() in traci.trafficlight.getIDList()):
                tls_count += 1
        except Exception as e:
            print(f"Error checking traffic lights for edge {edge_id}: {e}")
    return tls_count

#------------------------------------ trained data usage ------------------------------------------------------------------------

def find_path_between_edges(training_routes, start_edge, end_edge):
    
    #Find a path between specified start and end edges from training routes
    
    #Args:
     #   training_routes (List[List[str]]): List of previously generated routes
      #  start_edge (str): Starting edge ID
       # end_edge (str): Destination edge ID
    
    #Returns:
     #   List[str]: A valid path or None if no path found
    
    # Filter routes that start with the start_edge and end with the end_edge
    matching_routes = [
        route for route in training_routes 
        if route[0] == start_edge and route[-1] == end_edge
    ]
    
    if matching_routes:
        # Return the first matching route
        return matching_routes[0]
    
    # If no direct match, find routes with start and end edges
    candidate_routes = [
        route for route in training_routes 
        if start_edge in route and end_edge in route
    ]
    
    if candidate_routes:
        # Filter routes where start_edge comes before end_edge
        valid_routes = [
            route for route in candidate_routes
            if route.index(start_edge) < route.index(end_edge)
        ]
        
        # Return the first valid route if exists
        return valid_routes[0] if valid_routes else None
    
    return None

def random_edge_path_selection(net, training_routes):
    
   # Interactively select random edges and find a path from training routes
    
    #Args:
     #   net (sumolib.net.Net): SUMO network object
      #  training_routes (List[List[str]]): List of previously generated routes
    
    #Returns:
     #   List[str]: Selected path or None
    
    # Get all available edges
    all_edges = [edge.getID() for edge in net.getEdges()]
    
    while True:
        print("\n--- Random Edge Path Selection ---")
        print("Available Edges:")
        for i, edge in enumerate(all_edges, 1):
            print(f"{i}. {edge}")
        
        try:
            # Input for start edge
            start_index = int(input("\nEnter the number for START edge: ")) - 1
            start_edge = all_edges[start_index]
            
            # Input for destination edge
            end_index = int(input("Enter the number for END edge: ")) - 1
            end_edge = all_edges[end_index]
            
            # Validate input
            if start_edge != end_edge:
                # Find path between selected edges
                selected_path = find_path_between_edges(training_routes, start_edge, end_edge)
                
                if selected_path:
                    print("\nFound Path:")
                    print(" -> ".join(selected_path))
                    
                    # Calculate route metrics
                    route_length = sum(edge_lengths.get(edge, 0) for edge in selected_path)
                    vehicle_count = sum(traci.edge.getLastStepVehicleNumber(edge) for edge in selected_path)
                    traffic_lights = sum(predict_tls_count_sb3(edge) for edge in selected_path)
                    
                    print(f"\nRoute Metrics:")
                    print(f"Length: {route_length:.2f} meters")
                    print(f"Total Vehicles: {vehicle_count}")
                    print(f"Traffic Lights: {traffic_lights}")
                    
                    return selected_path
                else:
                    print("No path found between selected edges in training routes.")
                    
                    # Ask if user wants to try again
                    retry = input("Do you want to try again? (y/n): ").lower()
                    if retry != 'y':
                        break
            else:
                print("Start and destination edges must be different!")
        
        except (ValueError, IndexError):
            print("Invalid input. Please enter a valid edge number.")
    
    return None


# ------------------------------------------------------------------------------
# SUMO command setup (use "sumo-gui" for visualization if desired)
sumoBinary = "sumo"
sumo_cmd = [sumoBinary, "-c", sumocfg_file]

# ------------------------------------------------------------------------------
# Main execution block
if __name__ == "__main__":
    # Define start and destination edges (must exist in your SUMO network)
    start_edge = "E7"        # Replace with your actual start edge ID
    destination_edge = "E6"  # Replace with your actual destination edge ID

    # Collect training and test data
    X_train, Y_train = collect_training_data(start_edge, destination_edge, simulation_steps=TRAIN_SIM_STEPS)
    X_test, Y_test = collect_test_data(start_edge, destination_edge, simulation_steps=TEST_SIM_STEPS)
    
    # Train the CNN model if data is available
    if X_train.size and Y_train.size:
        cnn_model = train_cnn_model(X_train, Y_train, X_test, Y_test)
        # Use the new function to find the optimal path from start_edge to destination_edge
        optimal_path = find_optimal_path_all(cnn_model, start_edge, destination_edge)
        print("Final optimal route:", optimal_path)
        op= len(optimal_path)
        print("The edge number in the optimal path :", op)
        # # Calculate total distance in meters
        optimal_path_distance_m = sum(edge_lengths[edge] for edge in optimal_path)
         # Convert to kilometers if needed
        optimal_path_distance_km = optimal_path_distance_m / 1000.0
        print("Optimal path distance: {} meters".format(optimal_path_distance_m))
        print("Optimal path distance: {:.2f} km".format(optimal_path_distance_km))
        # Count and display the number of traffic lights in the optimal route
        num_traffic_lights = count_traffic_lights_in_path(optimal_path)
        print("Number of traffic lights in the optimal path:", num_traffic_lights)
        
        # Add option for random edge path selection
        while True:
            print("\n--- Post-Training Options ---")
            print("1. Select random edges path from training routes")
            print("2. Exit")
            
            choice = input("Enter your choice (1-2): ")
            
            if choice == '1':
                # Ensure TRAINING_ROUTES is populated and network is available
                if 'TRAINING_ROUTES' in locals() or 'TRAINING_ROUTES' in globals():
                    # Reuse the network object from earlier network reading
                    net = sumolib.net.readNet(net_file)
                    
                    selected_random_path = random_edge_path_selection(net, TRAINING_ROUTES)
                    if selected_random_path:
                        print("\nSelected Random Path Details processed successfully.")
                else:
                    print("No training routes available. Run training first.")
            
            elif choice == '2':
                print("Exiting program.")
                break

    else:
        print("Insufficient training data. Please check simulation settings and network connectivity.")
