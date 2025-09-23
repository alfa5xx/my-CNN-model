# Import necessary libraries for TensorFlow/Keras and SUMO simulation
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Import SUMO libraries
import traci
import sumolib
from collections import defaultdict

# ------------------------------------------------------------------------------
# Define file paths for the SUMO simulation files
net_file = "C:/Users/Administrator/Desktop/Prof.Mangini/Project2025/b.net.xml"      # Path to the SUMO network file
route_file = "C:/Users/Administrator/Desktop/Prof.Mangini/Project2025/b.rou.xml"    # Path to the routes file
sumocfg_file = "C:/Users/Administrator/Desktop/Prof.Mangini/Project2025/b.sumocfg"  # Path to the simulation configuration file

# ------------------------------------------------------------------------------
# Retrieve all edge IDs from the SUMO network using sumolib.
# This gives us the list of edges that a vehicle can follow.
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
# Start SUMO simulation using TraCI
sumoBinary = "sumo"  # use "sumo-gui" if you want visualization
sumo_cmd = [sumoBinary, "-c", sumocfg_file]

# ------------------------------------------------------------------------------
# Function to generate all possible paths (limited by max_depth for practical reasons)
def generate_all_paths(start_edge, destination_edge, max_depth=10):
    """Generate all possible paths from start_edge to destination_edge with max_depth limit."""
    all_paths = []
    
    def dfs(current_edge, current_path, visited):
        # If the path is too long or we're in a loop, stop exploration
        if len(current_path) > max_depth or current_edge in visited:
            return
        
        current_path.append(current_edge)
        visited.add(current_edge)
        
        # If we've reached the destination, add the path to our collection
        if current_edge == destination_edge:
            all_paths.append(list(current_path))
        else:
            # Continue exploring outgoing edges
            for next_edge in connectivity_map[current_edge]:
                dfs(next_edge, current_path, visited)
        
        # Backtrack
        current_path.pop()
        visited.remove(current_edge)
    
    # Start DFS from the start edge
    dfs(start_edge, [], set())
    print(f"DEBUG: Generated {len(all_paths)} paths from {start_edge} to {destination_edge}")
    return all_paths

# ------------------------------------------------------------------------------
# Feature extraction function to get real data from SUMO simulation
def extract_features(edge_id):
    """Extract features for a given edge from the SUMO simulation."""
    try:
        # Grid size for the feature representation
        grid_size = (64, 64)
        
        # 1. Vehicle density
        vehicles = traci.edge.getLastStepVehicleNumber(edge_id)
        edge_length = edge_lengths[edge_id]
        vehicle_density = vehicles / max(edge_length, 1)  # Vehicles per unit length
        
        # 2. Traffic lights
        tls_count = 0
        # Get lanes of this edge
        lanes = traci.edge.getLaneIDList(edge_id)
        for lane in lanes:
            # Get traffic lights linked to this lane
            tls = traci.lane.getLinks(lane)
            tls_count += sum(1 for link in tls if link[3])  # Count links with traffic lights
        
        # 3. Calculate edge speed factor (higher is better)
        current_speed = traci.edge.getLastStepMeanSpeed(edge_id)
        max_speed = traci.edge.getMaxSpeed(edge_id)
        speed_factor = current_speed / max_speed if max_speed > 0 else 0
        
        # Create feature grids - in a real scenario, this would be a spatial representation
        # For this example, we create fixed patterns based on the features
        vehicle_grid = np.ones(grid_size) * vehicle_density
        tls_grid = np.ones(grid_size) * (tls_count / max(len(lanes), 1))
        speed_grid = np.ones(grid_size) * speed_factor
        
        # Stack the channels
        return np.stack([vehicle_grid, tls_grid, speed_grid], axis=-1)
    except Exception as e:
        print(f"Error extracting features for edge {edge_id}: {e}")
        # Return fallback features
        return np.zeros((64, 64, 3))

# ------------------------------------------------------------------------------
# Path evaluation function
def evaluate_path(path):
    """Evaluate a path based on multiple factors."""
    try:
        if not path:
            return float('inf')  # Invalid path gets worst score
        
        total_length = sum(edge_lengths[edge] for edge in path)
        total_vehicles = sum(traci.edge.getLastStepVehicleNumber(edge) for edge in path)
        
        # Count traffic lights along the path
        total_tls = 0
        for edge in path:
            lanes = traci.edge.getLaneIDList(edge)
            for lane in lanes:
                links = traci.lane.getLinks(lane)
                total_tls += sum(1 for link in links if link[3])
        
        # Calculate overall score (lower is better)
        # We can adjust weights to prioritize different factors
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
# Data Collection Function:
# Generate training data from all possible paths
def collect_training_data(start_edge, destination_edge, simulation_steps=100):
    """
    Generate training data by evaluating multiple paths and selecting optimal segments.
    """
    X_data = []  # To hold feature matrices
    Y_data = []  # To hold labels (optimal next edge as one-hot vector)
    
    try:
        # Start SUMO
        traci.start(sumo_cmd)
        print("SUMO started for training data collection")
        
        # Generate all possible paths
        all_paths = generate_all_paths(start_edge, destination_edge)
        print(f"Generated {len(all_paths)} possible paths from {start_edge} to {destination_edge}")
        
        if not all_paths:
            print("No valid paths found. Check connectivity between start and destination.")
            traci.close()
            return np.array([]), np.array([])
        
        # For each simulation step, we evaluate paths in the current traffic conditions
        for step in range(simulation_steps):
            # Advance simulation
            traci.simulationStep()
            
            # Skip some steps if simulation is too dense to get varied traffic conditions
            if step % 5 != 0:
                continue
            
            print(f"Processing simulation step {step}/{simulation_steps}")
            
            # Evaluate all paths in current traffic conditions
            path_scores = []
            for path in all_paths:
                score = evaluate_path(path)
                path_scores.append((path, score))
            
            # Sort paths by score (lower is better)
            path_scores.sort(key=lambda x: x[1])
            best_paths = path_scores[:min(5, len(path_scores))]  # Get top 5 paths
            
            # For each position in these best paths, learn which edge to choose next
            for path, _ in best_paths:
                for i in range(len(path) - 1):
                    current_edge = path[i]
                    next_edge = path[i + 1]
                    
                    # Extract features for the current edge
                    features = extract_features(current_edge)
                    X_data.append(features)
                    
                    # Create one-hot vector for the next edge
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
    """Similar to collect_training_data but for test data"""
    X_test = []
    Y_test = []
    
    try:
        traci.start(sumo_cmd)
        print("SUMO started for test data collection")
        
        # Generate all possible paths
        all_paths = generate_all_paths(start_edge, destination_edge)
        
        if not all_paths:
            print("No valid paths found for testing.")
            traci.close()
            return np.array([]), np.array([])
        
        # Different traffic conditions for testing
        for step in range(simulation_steps):
            traci.simulationStep()
            
            if step % 2 != 0:
                continue
            
            print(f"Processing test simulation step {step}/{simulation_steps}")
            
            # Evaluate paths
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
    
    # Define CNN Architecture
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
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    # Train the model
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=15,
        batch_size=32
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
    plt.close()
    
    return model

# ------------------------------------------------------------------------------
# Function to find optimal path using the trained CNN model
def find_optimal_path(model, start_edge, destination_edge, max_steps=30):
    """
    Find the optimal path using the trained CNN model.
    Combines model predictions with connectivity constraints.
    """
    try:
        # Start SUMO
        traci.start(sumo_cmd)
        print("SUMO started for path finding")
        
        # Generate all possible paths for reference and validation
        all_possible_paths = generate_all_paths(start_edge, destination_edge)
        print(f"Reference: Found {len(all_possible_paths)} possible paths from {start_edge} to {destination_edge}")
        
        # Find the optimal path using the CNN model
        current_edge = start_edge
        optimal_path = [current_edge]
        visited = set([current_edge])
        
        for step in range(max_steps):
            # Stop if we've reached the destination
            if current_edge == destination_edge:
                break
            
            # Extract features for current edge
            features = extract_features(current_edge)
            input_features = np.expand_dims(features, axis=0)
            
            # Use the model to predict next edge probabilities
            predictions = model.predict(input_features)[0]
            \print (predictions)