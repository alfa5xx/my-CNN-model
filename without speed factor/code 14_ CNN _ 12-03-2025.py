# Import necessary libraries for TensorFlow/Keras and SUMO simulation
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Import SUMO libraries (ensure SUMO_HOME is set in your environment)
import traci
import sumolib



# Define file paths for the SUMO simulation files
net_file = "C:/Users/Administrator/Desktop/Prof.Mangini/Project2025/b.net.xml"      # Path to the SUMO network file
route_file = "C:/Users/Administrator/Desktop/Prof.Mangini/Project2025//b.rou.xml"       # Path to the routes file
sumocfg_file = "C:/Users/Administrator/Desktop/Prof.Mangini/Project2025//b.sumocfg" # Path to the simulation configuration file

# ------------------------------------------------------------------------------
# Start SUMO simulation using TraCI (choose "sumo" or "sumo-gui")
sumoBinary = "sumo"  # or "sumo-gui" for visualization
traci.start([sumoBinary, "-c", sumocfg_file])

# ------------------------------------------------------------------------------
# Data Collection Function:
# Run the simulation and extract input features as a 64x64 grid with 3 channels:
#   Channel 0: Vehicle density
#   Channel 1: Traffic lights status
#   Channel 2: Travel time / cost metric per edge
# For demonstration, we use random dummy data. Replace with your actual feature extraction.
def collect_data(simulation_steps=1000):
    grid_size = (64, 64)
    X_data = []  # List to hold feature matrices
    Y_data = []  # List to hold labels (optimal move as one-hot vectors)
    
    for step in range(simulation_steps):
        traci.simulationStep()  # Advance simulation by one step
        
        # Dummy feature extraction for each channel:
        vehicle_density = np.random.rand(*grid_size)
        traffic_lights  = np.random.rand(*grid_size)
        travel_time     = np.random.rand(*grid_size)
        
        input_matrix = np.stack([vehicle_density, traffic_lights, travel_time], axis=-1)
        X_data.append(input_matrix)
        
        # Dummy optimal move: generate a random one-hot label for 10 possible moves.
        num_moves = 10
        label = np.zeros(num_moves)
        optimal_move = np.random.randint(0, num_moves)
        label[optimal_move] = 1
        Y_data.append(label)
    
    traci.close()  # End simulation after data collection
    return np.array(X_data), np.array(Y_data)

# ------------------------------------------------------------------------------
# Collect training data from the SUMO simulation
X_train, Y_train = collect_data(simulation_steps=1000)
input_shape = X_train.shape[1:]  # Expected to be (64, 64, 3)

# ------------------------------------------------------------------------------
########### CNN Architecture Definition ###########
# Build the CNN model to learn the mapping between simulation features and optimal moves.
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
model.add(Dense(10, activation='softmax'))
########### End of CNN Architecture ###########

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ------------------------------------------------------------------------------
# Train the CNN using the training data extracted from the SUMO simulation.
model.fit(X_train, Y_train, epochs=10, batch_size=16)

# ------------------------------------------------------------------------------
# Define a mapping from predicted move indices to actual SUMO edge IDs.
# Replace these placeholder IDs with the actual IDs from your SUMO network.
edge_mapping = {
    0: "edge_A",
    1: "edge_B",
    2: "edge_C",
    3: "edge_D",
    4: "edge_E",
    5: "edge_F",
    6: "edge_G",
    7: "edge_H",
    8: "edge_I",
    9: "edge_J"
}

# ------------------------------------------------------------------------------
# Optimal Path Planning Function:
# This function uses the trained CNN to plan an optimal route from a given start edge to a destination edge.
# At each planning step the CNN predicts the next move which is then mapped to an edge ID.
# The loop stops when the destination is reached or when a loop (or max step count) is detected.
def plan_optimal_path(model, start_edge, destination_edge, max_steps=20):
    # For demonstration, we use dummy feature extraction at each step.
    # In a real application, extract features for the current state and current edge.
    optimal_path = [start_edge]
    current_edge = start_edge

    for step in range(max_steps):
        # Dummy feature extraction (replace with real data based on current_edge)
        vehicle_density = np.random.rand(64, 64)
        traffic_lights  = np.random.rand(64, 64)
        travel_time     = np.random.rand(64, 64)
        input_matrix = np.stack([vehicle_density, traffic_lights, travel_time], axis=-1)
        input_batch = np.expand_dims(input_matrix, axis=0)
        
        # Use the trained model to predict the next move.
        prediction = model.predict(input_batch)
        move = np.argmax(prediction)
        next_edge = edge_mapping[move]
        
        # Append next_edge if it is a new edge
        if next_edge in optimal_path:
            # Avoid loops: break if the next edge was already visited
            break
        optimal_path.append(next_edge)
        
        # Check if destination has been reached.
        if next_edge == destination_edge:
            break
        
        # Update the current edge for the next iteration.
        current_edge = next_edge

    print("Optimal Path (sequence of edge IDs):", optimal_path)

# ------------------------------------------------------------------------------
# Example of planning an optimal path:
# Specify your desired start and destination edge IDs.
start_edge = "edge_A"
destination_edge = "edge_E"
plan_optimal_path(model, start_edge, destination_edge, max_steps=20)
