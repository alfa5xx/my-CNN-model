import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import traci
import sumolib
import math

# ------------------------------------------------------------------------------
# Define file paths for the SUMO simulation files
net_file = "C:/Users/Administrator/Desktop/Prof.Mangini/Project2025/b.net.xml"      # Path to the SUMO network file
route_file = "C:/Users/Administrator/Desktop/Prof.Mangini/Project2025/b.rou.xml"       # Path to the routes file
sumocfg_file = "C:/Users/Administrator/Desktop/Prof.Mangini/Project2025/b.sumocfg" # Path to the simulation configuration file

# ------------------------------------------------------------------------------
# Load the SUMO network and retrieve all edge IDs using sumolib.
net = sumolib.net.readNet(net_file)
edge_list = [edge.getID() for edge in net.getEdges()]
n_edges = len(edge_list)
print("Retrieved edge IDs from SUMO map:", edge_list)

# Build a dictionary mapping each edge to its outgoing neighbor edge IDs.
edge_neighbors = {}
for edge in net.getEdges():
    neighbors = []
    # Each connection from the current edge
    for conn in edge.getConnections(None):  ####################### change to none
        # conn.getTo() returns the destination edge object.
        neighbors.append(conn.getTo().getID())
    edge_neighbors[edge.getID()] = neighbors

# ------------------------------------------------------------------------------
# Start SUMO simulation (for training data collection).
sumoBinary = "sumo"  # or "sumo-gui" for visualization
traci.start([sumoBinary, "-c", sumocfg_file])

# ------------------------------------------------------------------------------
# Data Collection Function:
# This function runs the simulation and extracts a dummy 64x64x3 feature grid.
# Each training label is a one-hot vector corresponding to one of the real edge IDs.
def collect_data(simulation_steps=1000):
    grid_size = (64, 64)
    X_data = []  # feature matrices
    Y_data = []  # one-hot labels for the optimal edge (dummy for demonstration)
    
    for step in range(simulation_steps):
        traci.simulationStep()
        # Dummy feature extraction. Replace with real traci data.
        vehicle_density = np.random.rand(*grid_size)
        traffic_lights  = np.random.rand(*grid_size)
        travel_time     = np.random.rand(*grid_size)
        input_matrix = np.stack([vehicle_density, traffic_lights, travel_time], axis=-1)
        X_data.append(input_matrix)
        
        # Dummy label: randomly choose one edge from the network.
        random_edge_index = np.random.randint(0, n_edges)
        label = np.zeros(n_edges)
        label[random_edge_index] = 1
        Y_data.append(label)
    
    traci.close()
    return np.array(X_data), np.array(Y_data)

# ------------------------------------------------------------------------------
# Collect training data from SUMO simulation.
X_train, Y_train = collect_data(simulation_steps=1000)
input_shape = X_train.shape[1:]  # Expected: (64, 64, 3)

# ------------------------------------------------------------------------------
########### CNN Architecture Definition ###########
# Build the CNN model. Its final Dense layer has n_edges neurons (one per edge).
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
########### End of CNN Architecture ###########

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ------------------------------------------------------------------------------
# Train the CNN.
model.fit(X_train, Y_train, epochs=10, batch_size=16)

# ------------------------------------------------------------------------------
# Dummy feature extraction function based on current edge.
# In a real system, this function would extract features from the SUMO state
# (e.g., vehicle density, traffic light status, travel time, etc.) near the given edge.
def get_features_for_edge(edge_id):
    # For demonstration, return a random 64x64x3 array.
    grid_size = (64, 64)
    vehicle_density = np.random.rand(*grid_size)
    traffic_lights  = np.random.rand(*grid_size)
    travel_time     = np.random.rand(*grid_size)
    return np.stack([vehicle_density, traffic_lights, travel_time], axis=-1)

# ------------------------------------------------------------------------------
# Beam Search Optimal Path Planning Function:
# This function explores multiple paths from the start to the destination edge.
# At each step it uses the CNN to score candidate next edges (restricted to valid neighbors)
# and expands the beam. The score is computed as the cumulative sum of log probabilities.
def search_optimal_path(model, start_edge, destination_edge, max_steps=20, beam_width=5):
    # Each beam entry is a tuple: (path, cumulative_score)
    # Start with the initial path.
    beams = [([start_edge], 0)]
    complete_paths = []

    for step in range(max_steps):
        new_beams = []
        for path, score in beams:
            current_edge = path[-1]
            # If current edge has no outgoing connections, skip expansion.
            if current_edge not in edge_neighbors or len(edge_neighbors[current_edge]) == 0:
                continue

            # Extract features for the current state (dummy function here).
            features = get_features_for_edge(current_edge)
            input_batch = np.expand_dims(features, axis=0)
            prediction = model.predict(input_batch)[0]  # vector of probabilities (length n_edges)

            # Consider only neighbors that are valid next moves.
            valid_neighbors = edge_neighbors[current_edge]
            for neighbor in valid_neighbors:
                # Find the global index for this neighbor.
                if neighbor in edge_list:
                    neighbor_index = edge_list.index(neighbor)
                    prob = prediction[neighbor_index]
                    # Avoid log(0) by using a small constant.
                    log_prob = math.log(prob + 1e-10)
                    new_score = score + log_prob
                    new_path = path + [neighbor]
                    # If destination reached, store complete path.
                    if neighbor == destination_edge:
                        complete_paths.append((new_path, new_score))
                    else:
                        new_beams.append((new_path, new_score))
        
        # If no new beams can be expanded, break out.
        if not new_beams:
            break

        # Sort new beams by cumulative score (higher is better) and keep top beam_width.
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_width]

    # If any complete paths were found, choose the one with the best (highest) score.
    if complete_paths:
        complete_paths.sort(key=lambda x: x[1], reverse=True)
        best_path, best_score = complete_paths[0]
        return best_path, best_score
    else:
        # Otherwise, return the best incomplete path.
        beams.sort(key=lambda x: x[1], reverse=True)
        return beams[0] if beams else (None, None)

# ------------------------------------------------------------------------------
# Define the start and destination edges.
# Replace these with the desired SUMO edge IDs.
start_edge = "E10"        # Define your desired start edge.
destination_edge = "E13"    # Define your desired destination edge.

# ------------------------------------------------------------------------------
# Execute the beam search to obtain the optimal full path.
best_path, best_score = search_optimal_path(model, start_edge, destination_edge, max_steps=20, beam_width=5)
if best_path:
    print("Optimal Full Path (edge IDs):", best_path)
    print("Cumulative Log Probability Score:", best_score)
else:
    print("No valid path found from", start_edge, "to", destination_edge)
