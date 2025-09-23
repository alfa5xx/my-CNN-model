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
TRAIN_EPOCHS = 100         # Number of epochs for training CNN
TRAIN_SIM_STEPS = 200      # Simulation steps for training data collection
TEST_SIM_STEPS = 50        # Simulation steps for test data collection
MODEL_CHOICE = "original"  # Options: "original" or "lightweight"
SHOW_PLOT = True           # If True, display training plot interactively

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
    """
    A custom environment for predicting the number of traffic lights on an edge.
    Observation: [number_of_lanes, average_lane_length]
    Action: Discrete value representing the predicted traffic light count (0 to 10)
    """
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
# For demonstration purposes, we initialize a PPO model.
# In practice, you would train this model on proper data or load a pre-trained model.
sb3_tls_model = PPO('MlpPolicy', env, verbose=0)
# To load a pre-trained model, uncomment the line below:
# sb3_tls_model = PPO.load("sb3_tls_model.zip")

# ------------------------------------------------------------------------------
# Function to predict the number of traffic lights on an edge using the SB3 model
def predict_tls_count_sb3(edge_id):
    """
    Predict the number of traffic lights on a given edge using the SB3 model.
    The observation is composed of the number of lanes and the average lane length.
    """
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
    #    Extract features for a given edge from the SUMO simulation.
   # Returns a 3-channel image where each channel is a grid (64x64) with:
    #- Channel 1: vehicle density (normalized)
    #- Channel 2: traffic light count (normalized)
    #- Channel 3: speed factor
    
    try:
        grid_size = (64, 64)
        # 1. Vehicle density
        vehicles = traci.edge.getLastStepVehicleNumber(edge_id)
        edge_length = edge_lengths[edge_id]
        vehicle_density = vehicles / max(edge_length, 1)
        # Normalize vehicle density to [0,1] with a reasonable max value
        max_expected_density = 0.5  # Adjust based on your simulation's characteristics
        vehicle_density = min(vehicle_density / max_expected_density, 1.0)
        
        # 2. Traffic lights count using SB3 model prediction
        predicted_tls = predict_tls_count_sb3(edge_id)
        edge_obj = net.getEdge(edge_id)
        lanes = [lane.getID() for lane in edge_obj.getLanes()]
        # Normalize traffic light value to [0,1]
        max_expected_tls = 10  # Maximum value from the SB3 model's action space
        tls_value = predicted_tls / max_expected_tls
        
        # 3. Speed factor (current speed divided by max speed)
        current_speed = traci.edge.getLastStepMeanSpeed(edge_id)
        max_speed = edge_obj.speed
        speed_factor = current_speed / max_speed if max_speed > 0 else 0
        
        vehicle_grid = np.ones(grid_size) * vehicle_density
        tls_grid = np.ones(grid_size) * tls_value
        speed_grid = np.ones(grid_size) * speed_factor
        
        return np.stack([vehicle_grid, tls_grid, speed_grid], axis=-1)
    except Exception as e:
        print(f"Error extracting features for edge {edge_id}: {e}")
        return np.zeros((64, 64, 3))
    
#----------------- new normalaize  code --

def prepare_normalized_data(X_train, Y_train, X_test, Y_test):
    """
    Normalize the input data globally across all samples.
    """
    if len(X_train) == 0 or len(Y_train) == 0:
        print("No training data available.")
        return None, None, None, None
    
    # Reshape to combine all spatial dimensions for normalization
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    orig_shape = X_train.shape[1:]
    
    X_train_flat = X_train.reshape(n_train, -1)
    X_test_flat = X_test.reshape(n_test, -1)
    
    # Calculate mean and std for each feature across all samples
    mean = np.mean(X_train_flat, axis=0)
    std = np.std(X_train_flat, axis=0)
    std[std == 0] = 1  # Prevent division by zero
    
    # Normalize both training and test data
    X_train_norm = (X_train_flat - mean) / std
    X_test_norm = (X_test_flat - mean) / std
    
    # Reshape back to original shape
    X_train_norm = X_train_norm.reshape(n_train, *orig_shape)
    X_test_norm = X_test_norm.reshape(n_test, *orig_shape)
    
    print("Data normalized with mean subtraction and standard deviation scaling")
    return X_train_norm, Y_train, X_test_norm, Y_test

    
    
    
  #-----------------------------  

# ------------------------------------------------------------------------------
# Path evaluation function with SB3 integration for traffic lights count
def evaluate_path(path):
    
    #Evaluate a path based on normalized metrics for total distance, vehicle count, 
    #traffic lights, and edge count.
    
    try:
        if not path:
            return float('inf')
        
        # Calculate raw values
        total_length = sum(edge_lengths[edge] for edge in path)
        total_vehicles = sum(traci.edge.getLastStepVehicleNumber(edge) for edge in path)
        total_tls = sum(predict_tls_count_sb3(edge) for edge in path)
        
        # Normalize values based on expected maximums
        # These max values should be adjusted based on your specific network
        max_expected_length = 10000  # Maximum expected path length
        max_expected_vehicles = 500  # Maximum expected vehicle count
        max_expected_tls = 50       # Maximum expected traffic light count
        max_expected_edges = 30      # Maximum expected edge count
        
        norm_length = total_length / max_expected_length
        norm_vehicles = total_vehicles / max_expected_vehicles
        norm_tls = total_tls / max_expected_tls
        norm_edge_count = len(path) / max_expected_edges
        
        # Weights for each component
        length_weight = 0.3
        vehicle_weight = 0.3
        tls_weight = 0.2
        edge_count_weight = 0.2
        
        # Calculate weighted score
        score = (length_weight * norm_length +
                 vehicle_weight * norm_vehicles +
                 tls_weight * norm_tls +
                 edge_count_weight * norm_edge_count)
        
        return score
    except Exception as e:
        print(f"Error evaluating path: {e}")
        return float('inf')
# ------------------------------------------------------------------------------
# Function to generate all possible paths (limited by max_depth for practical reasons)
def generate_all_paths(start_edge, destination_edge, max_depth=20):
    """
    Generate all possible paths from start_edge to destination_edge with a max_depth limit.
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

# ------------------------------------------------------------------------------
# Data Collection Functions:
def collect_training_data(start_edge, destination_edge, simulation_steps=TRAIN_SIM_STEPS):
    """
    Generate training data by evaluating multiple paths and selecting optimal segments.
    Returns feature matrices (X_data) and one-hot labels for the next edge (Y_data).
    """
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
# Define CNN model architectures with BatchNormalization to help training
def build_original_model(input_shape):
    model = Sequential()
    # Add a normalization layer to scale inputs
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
    
    # Normalize the data
    X_train_norm, Y_train, X_test_norm, Y_test = prepare_normalized_data(X_train, Y_train, X_test, Y_test)
    
    print(f"Training CNN model with {len(X_train_norm)} samples")
    print(f"Input shape: {X_train_norm.shape}, Output shape: {Y_train.shape}")
    input_shape = X_train_norm.shape[1:]
    
    if MODEL_CHOICE == "lightweight":
        print("Using lightweight CNN model for faster training.")
        model = build_lightweight_model(input_shape)
    else:
        print("Using original CNN model.")
        model = build_original_model(input_shape)
    
    # Using Adam optimizer with default learning rate; you can adjust if needed.
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
    
    history = model.fit(
        X_train_norm, Y_train,
        validation_data=(X_test_norm, Y_test),
        epochs=TRAIN_EPOCHS,
        batch_size=64,
        callbacks=[early_stop, reduce_lr]
    )
    
    # Plot training history and display interactively if requested
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
    
    return model, X_train_norm  # Return the normalized data for use in prediction
# ------------------------------------------------------------------------------
# Function to find the optimal path using the trained CNN model
def find_optimal_path(model, start_edge, destination_edge, max_steps=30, scaler=None):
    """
    Find the optimal path using the trained CNN model with normalized inputs.
    """
    try:
        traci.start(sumo_cmd)
        print("SUMO started for path finding")
        current_edge = start_edge
        optimal_path = [current_edge]
        visited = set([current_edge])
        
        # Store normalization parameters
        norm_params = {}
        
        for step in range(max_steps):
            if current_edge == destination_edge:
                break
            
            # Extract features and normalize them
            features = extract_features(current_edge)
            input_features = np.expand_dims(features, axis=0)
            
            # Apply the same normalization as during training
            # This is a simplified approach - ideally you'd use the same scaler from training
            if scaler:
                input_features = scaler.transform(input_features.reshape(1, -1)).reshape(input_features.shape)
            
            predictions = model.predict(input_features)[0]
            possible_next_edges = connectivity_map[current_edge]
            
            if destination_edge in possible_next_edges:
                optimal_path.append(destination_edge)
                print(f"Destination {destination_edge} is directly accessible from {current_edge}.")
                break
            
            candidate_scores = {}
            for candidate in possible_next_edges:
                if candidate in visited:
                    continue
                    
                cnn_score = 1 - predictions[edge_to_index[candidate]]
                simulated_path = optimal_path + [candidate]
                eval_score = evaluate_path(simulated_path)
                combined_score = 0.5 * cnn_score + 0.5 * eval_score
                candidate_scores[candidate] = combined_score
            
            if not candidate_scores:
                print("No valid candidates found. Ending path search.")
                break
                
            selected_edge = min(candidate_scores, key=candidate_scores.get)
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
# SUMO command setup (use "sumo-gui" for visualization if desired)
sumoBinary = "sumo"
sumo_cmd = [sumoBinary, "-c", sumocfg_file]

# ------------------------------------------------------------------------------
# Main execution block
if __name__ == "__main__":
    # Define start and destination edges (must exist in your SUMO network)
    start_edge = "E7" #"E10"        # Replace with your actual start edge ID
    destination_edge = "E8" #"E13"  # Replace with your actual destination edge ID

    # Collect training and test data
    X_train, Y_train = collect_training_data(start_edge, destination_edge, simulation_steps=TRAIN_SIM_STEPS)
    X_test, Y_test = collect_test_data(start_edge, destination_edge, simulation_steps=TEST_SIM_STEPS)
    
    # Train the CNN model if data is available
    if X_train.size and Y_train.size:
        cnn_model = train_cnn_model(X_train, Y_train, X_test, Y_test)
        # Find the optimal path after training
        optimal_path = find_optimal_path(cnn_model, start_edge, destination_edge, max_steps=40)
        print("Final optimal route:", optimal_path)
    else:
        print("Insufficient training data. Please check simulation settings and network connectivity.")
