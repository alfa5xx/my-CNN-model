import os
import numpy as np
import tensorflow as tf
import networkx as nx
import traci
import sumolib
import math

# ------------------------------------------------------------------------------
# Define file paths for the SUMO simulation files
net_file = "C:/Users/Administrator/Desktop/Prof.Mangini/Project2025/b.net.xml"      # Path to the SUMO network file
route_file = "C:/Users/Administrator/Desktop/Prof.Mangini/Project2025/b.rou.xml"       # Path to the routes file
sumocfg_file = "C:/Users/Administrator/Desktop/Prof.Mangini/Project2025/b.sumocfg" # Path to the simulation configuration file

# ------------------------------------------------------------------------------
# Load the SUMO network using sumolib.
net = sumolib.net.readNet(net_file)


# Create a NetworkX graph from the SUMO network.
G = nx.DiGraph()
for edge in net.getEdges():
    edge_id = edge.getID()
    from_node = edge.getFromNode().getID()
    to_node = edge.getToNode().getID()
    length = edge.getLength()
    speed = edge.getSpeed()
    # Add the edge with its attributes to the graph
    G.add_edge(from_node, to_node, id=edge_id, length=length, speed=speed)

# ------------------------------------------------------------------------------
# Start SUMO simulation (for training data collection).
sumoBinary = "sumo"  # or "sumo-gui" for visualization
traci.start([sumoBinary, "-c", sumocfg_file])

# ------------------------------------------------------------------------------
# Data Collection Function:
# This function runs the simulation and extracts features for each edge.
def collect_data(simulation_steps=1000):
    edge_features = {}
    for step in range(simulation_steps):
        traci.simulationStep()
        for edge in net.getEdges():
            # Extract features for each edge.
            # Replace with real feature extraction logic.
            vehicle_density = np.random.rand()
            traffic_lights  = np.random.rand()
            travel_time     = np.random.rand()
            edge_features[edge.getID()] = [vehicle_density, traffic_lights, travel_time]
    traci.close()
    return edge_features

# ------------------------------------------------------------------------------
# Collect edge features from SUMO simulation.
edge_features = collect_data(simulation_steps=1000)

# ------------------------------------------------------------------------------
# Load pre-trained CNN model.
# Ensure the model is trained to accept edge feature vectors and output a single scalar score.
model = tf.keras.models.load_model('path_to_your_trained_model.h5')

# ------------------------------------------------------------------------------
# Function to extract features for a given path.
def extract_path_features(path, edge_features):
    features = []
    for edge_id in path:
        features.append(edge_features.get(edge_id, [0, 0, 0]))  # Default to zeros if edge not found
    return np.array(features)

# ------------------------------------------------------------------------------
# Function to find all paths between start and end edges.
def find_all_paths(G, start, end, path=[]):
    path = path + [start]
    if start == end:
        yield path
    for neighbor in G.neighbors(start):
        if neighbor not in path:
            yield from find_all_paths(G, neighbor, end, path)

# ------------------------------------------------------------------------------
# Define the start and destination edges.
start_edge = "E10"        # Define your desired start edge.
destination_edge = "E13"    # Define your desired destination edge.

# ------------------------------------------------------------------------------
# Generate all possible paths from start_edge to destination_edge.
all_paths = list(find_all_paths(G, start_edge, destination_edge))

# ------------------------------------------------------------------------------
# Evaluate each path using the CNN and select the optimal path.
best_path = None
best_score = -float('inf')
for path in all_paths:
    path_features = extract_path_features(path, edge_features)
    path_features = np.expand_dims(path_features, axis=0)  # Add batch dimension
    score = model.predict(path_features)[0][0]  # Assuming model outputs a single scalar score
    if score > best_score:
        best_score = score
        best_path = path

# ------------------------------------------------------------------------------
# Output the optimal path.
if best_path:
    print("Optimal Full Path (edge IDs):", best_path)
    print("CNN Score:", best_score)
else:
    print("No valid path found from", start_edge, "to", destination_edge)
