# Import necessary libraries for TensorFlow/Keras and SUMO simulation
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Import SUMO libraries
import traci
import sumolib

# ------------------------------------------------------------------------------
# Define file paths for the SUMO simulation files
net_file = "C:/Users/Administrator/Desktop/Prof.Mangini/Project2025/b.net.xml"      # Path to the SUMO network file
route_file = "C:/Users/Administrator/Desktop/Prof.Mangini/Project2025/b.rou.xml"       # Path to the routes file
sumocfg_file = "C:/Users/Administrator/Desktop/Prof.Mangini/Project2025/b.sumocfg" # Path to the simulation configuration file

# ------------------------------------------------------------------------------
# Retrieve all edge IDs from the SUMO network using sumolib.
# This gives us the list of edges that a vehicle can follow.
net = sumolib.net.readNet(net_file)
edge_list = [edge.getID() for edge in net.getEdges()]
n_edges = len(edge_list)
print("Retrieved edge IDs from SUMO map:", edge_list)

# ------------------------------------------------------------------------------
# Start SUMO simulation using TraCI (choose "sumo" or "sumo-gui")
sumoBinary = "sumo"  # use "sumo-gui" if you want visualization
traci.start([sumoBinary, "-c", sumocfg_file])

# ------------------------------------------------------------------------------
# Data Collection Function:
# This function runs the simulation for a number of steps, extracts features from
# the simulation, and assigns a training label. Here, each label is a one-hot vector
# of length n_edges (the number of edges in the network). For demonstration, we randomly
# select one edge as the "optimal" move at each step.
def collect_data(simulation_steps=1000):
    grid_size = (64, 64)  # Dimensions for our feature grid
    X_data = []  # To hold feature matrices
    Y_data = []  # To hold labels (optimal edge as one-hot vector)
    
    for step in range(simulation_steps):
        traci.simulationStep()  # Advance simulation by one step
        
        # --- Feature Extraction ---
        # Dummy feature extraction: In practice, replace these with actual data from traci.
        vehicle_density = np.random.rand(*grid_size)
        traffic_lights  = np.random.rand(*grid_size)
        travel_time     = np.random.rand(*grid_size)
        # Stack channels to create a (64, 64, 3) input matrix.
        input_matrix = np.stack([vehicle_density, traffic_lights, travel_time], axis=-1)
        X_data.append(input_matrix)
        
        # --- Label Generation ---
        # For demonstration, randomly choose one edge (from the full list) as the optimal move.
        random_edge_index = np.random.randint(0, n_edges)
        label = np.zeros(n_edges)
        label[random_edge_index] = 1
        Y_data.append(label)
    
    traci.close()  # End simulation after data collection
    return np.array(X_data), np.array(Y_data)

# ------------------------------------------------------------------------------
# Collect training data from the SUMO simulation.
X_train, Y_train = collect_data(simulation_steps=1000)
input_shape = X_train.shape[1:]  # Expected shape: (64, 64, 3)

# ------------------------------------------------------------------------------
########### CNN Architecture Definition ###########
# Build the CNN model to learn the mapping between simulation features and the optimal edge.
# The final output layer's dimension is set to n_edges, corresponding to the full list of edge IDs.
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

# Compile the CNN model.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ------------------------------------------------------------------------------
# Train the CNN using the training data extracted from the SUMO simulation.
model.fit(X_train, Y_train, epochs=10, batch_size=16)

# ------------------------------------------------------------------------------
# Optimal Path Planning Function:
# This function uses the trained CNN to plan an optimal route from a given start edge
# to a destination edge. At each planning step, the CNN predicts the next edge (using the
# full list of edge IDs retrieved earlier). Loop avoidance is implemented to prevent revisiting.
def plan_optimal_path(model, start_edge, destination_edge, max_steps=20):
    # Restart the simulation for planning.
    traci.start([sumoBinary, "-c", sumocfg_file])
    optimal_path = [start_edge]
    current_edge = start_edge

    for step in range(max_steps):
        # --- Feature Extraction for Current State ---
        # Replace these dummy features with actual state features based on current_edge.
        vehicle_density = np.random.rand(64, 64)
        traffic_lights  = np.random.rand(64, 64)
        travel_time     = np.random.rand(64, 64)
        input_matrix = np.stack([vehicle_density, traffic_lights, travel_time], axis=-1)
        input_batch = np.expand_dims(input_matrix, axis=0)
        
        # --- Predict the Next Edge ---
        prediction = model.predict(input_batch)
        move = np.argmax(prediction)
        next_edge = edge_list[move]  # Map prediction index to actual edge ID
        
        # --- Loop Avoidance: Break if next_edge already exists in path ---
        if next_edge in optimal_path:
            break
        
        optimal_path.append(next_edge)
        
        # --- Check for Destination ---
        if next_edge == destination_edge:
            break
        
        current_edge = next_edge

    traci.close()
    print("Optimal Path (edge IDs):", optimal_path)

# ------------------------------------------------------------------------------
# Example of planning an optimal path:
# Specify your start and destination edge IDs (must be from edge_list).
start_edge = edge_list[0]         # For example, the first edge in the list.
destination_edge = edge_list[-1]    # For example, the last edge in the list.
plan_optimal_path(model, start_edge, destination_edge, max_steps=20)
