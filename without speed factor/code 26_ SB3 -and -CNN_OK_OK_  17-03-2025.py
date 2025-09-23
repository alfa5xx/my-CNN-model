# Import necessary libraries for TensorFlow/Keras, Stable Baselines3, Gym, and SUMO simulation
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
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
# Define file paths for the SUMO simulation files
net_file = "C:/Users/Administrator/Desktop/Prof.Mangini/Project2025/b.net.xml"      # SUMO network file
route_file = "C:/Users/Administrator/Desktop/Prof.Mangini/Project2025/b.rou.xml"    # Routes file
sumocfg_file = "C:/Users/Administrator/Desktop/Prof.Mangini/Project2025/b.sumocfg"  # Simulation configuration file

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
   # Observation: [number_of_lanes, average_lane_length]
   # Action: Discrete value representing the predicted traffic light count (0 to 10)
    
    def __init__(self):
        super(TrafficLightEnv, self).__init__()
        # Define observation space: number of lanes and average lane length (range is arbitrary)
        self.observation_space = spaces.Box(low=0, high=1000, shape=(2,), dtype=np.float32)
        # Define action space: predicted number of traffic lights (0 to 10)
        self.action_space = spaces.Discrete(11)
        
    def reset(self):
        # For prediction, reset returns a default observation.
        return np.zeros(2, dtype=np.float32)
    
    def step(self, action):
        # Dummy step function (not used for prediction)
        observation = np.zeros(2, dtype=np.float32)
        reward = 0.0
        done = True
        info = {}
        return observation, reward, done, info

# Initialize the SB3 environment and model for traffic light prediction.
env = TrafficLightEnv()
# For demonstration purposes, we initialize a PPO model.
# In practice, you would train this model on proper data or load a pre-trained model.
sb3_tls_model = PPO('MlpPolicy', env, verbose=0)
# Optionally, you could load a pre-trained model:
# sb3_tls_model = PPO.load("sb3_tls_model.zip")

# ------------------------------------------------------------------------------
# Function to predict the number of traffic lights on an edge using the SB3 model
def predict_tls_count_sb3(edge_id):
    
    #Predict the number of traffic lights on a given edge using the SB3 model.
    #The observation is composed of the number of lanes and the average lane length.
    
    try:
        edge_obj = net.getEdge(edge_id)
        lanes = edge_obj.getLanes()
        num_lanes = len(lanes)
        if num_lanes == 0:
            return 0
        lane_lengths = [lane.getLength() for lane in lanes]
        avg_lane_length = np.mean(lane_lengths)
        # Create observation vector
        obs = np.array([num_lanes, avg_lane_length], dtype=np.float32).reshape(1, -1)
        # Use the SB3 model to predict the action (traffic light count)
        action, _ = sb3_tls_model.predict(obs, deterministic=True)
        predicted_tls = int(action)
        return predicted_tls
    except Exception as e:
        print(f"Error in SB3 TLS prediction for edge {edge_id}: {e}")
        return 0

# ------------------------------------------------------------------------------
# Feature extraction function to get real data from SUMO simulation, with SB3 integration
def extract_features(edge_id):
    # --- Extract features for a given edge from the SUMO simulation."""
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
        
        # 3. Speed factor (current speed divided by max speed)
        current_speed = traci.edge.getLastStepMeanSpeed(edge_id)
        max_speed = edge_obj.speed
        speed_factor = current_speed / max_speed if max_speed > 0 else 0
        
        # Create feature grids based on the above metrics
        vehicle_grid = np.ones(grid_size) * vehicle_density
        tls_grid = np.ones(grid_size) * tls_value
        speed_grid = np.ones(grid_size) * speed_factor
        
        # Stack the channels together
        return np.stack([vehicle_grid, tls_grid, speed_grid], axis=-1)
    except Exception as e:
        print(f"Error extracting features for edge {edge_id}: {e}")
        return np.zeros((64, 64, 3))

# ------------------------------------------------------------------------------
# Path evaluation function with SB3 integration for traffic lights count
def evaluate_path(path):
   # -- Evaluate a path based on multiple factors: total distance, vehicle count, traffic lights, and edge count."""
    try:
        if not path:
            return float('inf')
        
        total_length = sum(edge_lengths[edge] for edge in path)
        total_vehicles = sum(traci.edge.getLastStepVehicleNumber(edge) for edge in path)
        
        # Count traffic lights along the path using SB3 prediction
        total_tls = 0
        for edge in path:
            predicted_tls = predict_tls_count_sb3(edge)
            total_tls += predicted_tls
        
        # Weights for each factor can be tuned as needed
        length_weight = 0.3
        vehicle_weight = 0.3
        tls_weight = 0.2
        edge_count_weight = 0.2
        
        score = (
            length_weight * total_length +
            vehicle_weight * total_vehicles +
            tls_weight * total_tls +
            edge_count_weight * len(path)
        )
        return score
    except Exception as e:
        print(f"Error evaluating path: {e}")
        return float('inf')

# ------------------------------------------------------------------------------
# Function to generate all possible paths (limited by max_depth for practical reasons)
def generate_all_paths(start_edge, destination_edge, max_depth=20):
    #----- Generate all possible paths from start_edge to destination_edge with a max_depth limit."""
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

# ------------------------------------------------------------------------------
# Data Collection Function:
def collect_training_data(start_edge, destination_edge, simulation_steps=100):
    
    # Generate training data by evaluating multiple paths and selecting optimal segments.
    #Returns feature matrices (X_data) and one-hot labels for the next edge (Y_data).
    
    X_data = []
    Y_data = []
    
    try:
        traci.start(sumo_cmd)
        print("SUMO started for training data collection")
        
        all_paths = generate_all_paths(start_edge, destination_edge)
        print(f"Generated {len(all_paths)} possible paths from {start_edge} to {destination_edge}")
        
        if not all_paths:
            print("No valid paths found. Check connectivity between start and destination.")
            traci.close()
            return np.array([]), np.array([])
        
        for step in range(simulation_steps):
            traci.simulationStep()
            if step % 5 != 0:
                continue
            
            print(f"Processing simulation step {step}/{simulation_steps}")
            
            path_scores = []
            for path in all_paths:
                score = evaluate_path(path)
                path_scores.append((path, score))
            
            # Sort paths by score (lower is better) and pick top 5
            path_scores.sort(key=lambda x: x[1])
            best_paths = path_scores[:min(5, len(path_scores))]
            
            for path, _ in best_paths:
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

# ------------------------------------------------------------------------------
# Function to collect test data
def collect_test_data(start_edge, destination_edge, simulation_steps=20):
    #--- Collect test data similar to training data collection.
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
            
            print(f"Processing test simulation step {step}/{simulation_steps}")
            path_scores = [(path, evaluate_path(path)) for path in all_paths]
            path_scores.sort(key=lambda x: x[1])
            best_paths = path_scores[:min(3, len(path_scores))]
            
            for path, _ in best_paths:
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
# Define and train the CNN model
def train_cnn_model(X_train, Y_train, X_test, Y_test):
    if len(X_train) == 0 or len(Y_train) == 0:
        print("No training data available.")
        return None
    
    print(f"Training CNN model with {len(X_train)} samples")
    print(f"Input shape: {X_train.shape}")
    print(f"Output shape: {Y_train.shape}")
    
    input_shape = X_train.shape[1:]  # Expected shape: (64, 64, 3)
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
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
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    # Define callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
    
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=150,
        batch_size=64,
        callbacks=[early_stop, reduce_lr]
    )
    
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
    plt.close()
    
    return model

# ------------------------------------------------------------------------------
# Function to find the optimal path using the trained CNN model
def find_optimal_path(model, start_edge, destination_edge, max_steps=30):
    #
    #Find the optimal path using the trained CNN model.
    #At each step, the model predicts next-edge probabilities.
    #For each candidate edge reachable from the current edge (based on connectivity),
   # a combined score is computed by mixing the CNN prediction and the evaluation of the simulated partial path.
    #
    try:
        traci.start(sumo_cmd)
        print("SUMO started for path finding")
        
        current_edge = start_edge
        optimal_path = [current_edge]
        visited = set([current_edge])
        
        for step in range(max_steps):
            if current_edge == destination_edge:
                break
            
            # Extract features for the current edge and predict next-edge probabilities
            features = extract_features(current_edge)
            input_features = np.expand_dims(features, axis=0)
            predictions = model.predict(input_features)[0]
            
            possible_next_edges = connectivity_map[current_edge]
            
            # If destination is directly reachable, choose it immediately.
            if destination_edge in possible_next_edges:
                optimal_path.append(destination_edge)
                print(f"Destination {destination_edge} is directly accessible from {current_edge}.")
                break   
            
            candidate_scores = {}
            for candidate in possible_next_edges:
                # Convert CNN prediction to a score (higher probability gives lower score)
                cnn_score = 1 - predictions[edge_to_index[candidate]]
                # Evaluate the candidate by simulating the partial path
                simulated_path = optimal_path + [candidate]
                eval_score = evaluate_path(simulated_path)
                # Combine the two scores (weights can be tuned; here they are equal)
                combined_score = 0.5 * cnn_score + 0.5 * eval_score
                candidate_scores[candidate] = combined_score
            
            # Choose the candidate with the minimum combined score
            selected_edge = min(candidate_scores, key=candidate_scores.get)
            if selected_edge in visited:
                print("Cycle detected. Ending path search.")
                break
            
            optimal_path.append(selected_edge)
            visited.add(selected_edge)
            current_edge = selected_edge
            print(f"Step {step+1}: Moved to edge {current_edge} with combined score {candidate_scores[selected_edge]:.4f}")
        
        traci.close()
        print(f"Optimal path found: {optimal_path}")
        return optimal_path
    except Exception as e:
        print(f"Error finding optimal path: {e}")
        if traci.isLoaded():
            traci.close()
        return []

# ------------------------------------------------------------------------------
# SUMO command setup
sumoBinary = "sumo"  # Use "sumo-gui" if you want visualization
sumo_cmd = [sumoBinary, "-c", sumocfg_file]

# ------------------------------------------------------------------------------
# Example usage:
if __name__ == "__main__":
    # Define the start and destination edges (should exist in your SUMO network)
    start_edge = "E7" #"E10"        # Replace with your actual start edge ID
    destination_edge = "E8"  #"E13"  # Replace with your actual destination edge ID

    # Collect training and test data
    X_train, Y_train = collect_training_data(start_edge, destination_edge, simulation_steps=200) # default value was  200
    X_test, Y_test = collect_test_data(start_edge, destination_edge, simulation_steps=40)
    
    # Train the CNN model if training data is available
    if X_train.size and Y_train.size:
        cnn_model = train_cnn_model(X_train, Y_train, X_test, Y_test)
    
        # After training, find the optimal path
        optimal_path = find_optimal_path(cnn_model, start_edge, destination_edge, max_steps=40)
        print("Final optimal route:", optimal_path)
    else:
        print("Insufficient training data. Please check simulation settings and network connectivity.")


#---------"""Explanation----------------------
#1.SB3 Integration for Traffic Lights:
#A custom Gym environment (TrafficLightEnv) is defined so that an SB3 PPO agent can be initialized. The new function predict_tls_count_sb3(edge_id) creates an observation based on the number of lanes and their average length and then calls the PPO model’s predict method to obtain a predicted count of traffic lights.

#2.Feature Extraction & Evaluation Updates:
#In both extract_features and evaluate_path, the original code that looped through lanes and used traci.lane.getLinks to count traffic lights is replaced by calls to predict_tls_count_sb3. This SB3-based prediction is then used to generate the “traffic lights” channel in the CNN’s input as well as in the path evaluation score.

#3.Preserved Structure:
#All remaining functionality—from generating paths, collecting training and test data, training the CNN model, to finding the optimal path—remains intact.

#Feel free to adjust the SB3 environment, training procedures, or reward design as needed for your specific application and available training data.