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
# Data Collection Function: Run the simulation and extract input features.
# In this example, we create a 64x64 grid with 3 channels for:
#   Channel 0: Vehicle density
#   Channel 1: Traffic lights status (e.g., active/inactive or count)
#   Channel 2: Travel time / cost metric per edge
# Replace dummy data with your actual extraction from the SUMO simulation.
def collect_data(simulation_steps=1000):
    grid_size = (64, 64)  # Define the dimensions of your input representation
    channels = 3          # Number of features/channels
    X_data = []           # List to hold feature matrices
    Y_data = []           # List to hold labels (e.g., optimal moves)
    
    for step in range(simulation_steps):
        # Advance simulation by one step
        traci.simulationStep()
        
        # ---------------------------------------------------------
        # Extract features from simulation:
        # Channel 0: Create a vehicle density map
        # (In practice, use traci.vehicle.getIDList() and positions to map vehicles to the grid.)
        vehicle_density = np.random.rand(*grid_size)
        
        # Channel 1: Create a traffic lights status map
        # (In practice, use traci.trafficlight.getIDList() and their state to fill the grid.)
        traffic_lights = np.random.rand(*grid_size)
        
        # Channel 2: Create a travel time / cost map
        # (In practice, compute the cost based on the network data or simulation outputs.)
        travel_time = np.random.rand(*grid_size)
        
        # Stack the channels together to form a single input matrix for this simulation step
        input_matrix = np.stack([vehicle_density, traffic_lights, travel_time], axis=-1)
        X_data.append(input_matrix)
        
        # ---------------------------------------------------------
        # Determine the optimal move/decision at the current state.
        # For demonstration, we generate a random one-hot encoded label.
        # Replace this with your logic for determining the optimal next step.
        num_moves = 10  # Assume there are 10 possible actions/edges
        label = np.zeros(num_moves)
        optimal_move = np.random.randint(0, num_moves)
        label[optimal_move] = 1
        Y_data.append(label)
    
    # End the simulation and close the TraCI connection
    traci.close()
    
    return np.array(X_data), np.array(Y_data)

# ------------------------------------------------------------------------------
# Collect training data from the SUMO simulation
X_train, Y_train = collect_data(simulation_steps=1000)

# Determine the input shape for the CNN based on collected data
input_shape = X_train.shape[1:]  # Expected to be (64, 64, 3)

# ------------------------------------------------------------------------------
########### CNN Architecture Definition ###########
# Build the CNN model to learn the mapping between SUMO simulation features and optimal moves.
model = Sequential()

# Convolutional Layer 1: 32 filters, 3x3 kernel, ReLU activation
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
# Pooling Layer 1: Reduce spatial dimensions
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional Layer 2: 64 filters, 3x3 kernel, ReLU activation
model.add(Conv2D(64, (3, 3), activation='relu'))
# Pooling Layer 2
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional Layer 3: 128 filters, 3x3 kernel, ReLU activation
model.add(Conv2D(128, (3, 3), activation='relu'))
# Pooling Layer 3
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the feature maps to a 1D vector
model.add(Flatten())

# Fully Connected Hidden Layer 1: 256 neurons with ReLU activation
model.add(Dense(256, activation='relu'))
# Dropout for regularization
model.add(Dropout(0.5))

# Fully Connected Hidden Layer 2: 128 neurons with ReLU activation
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Output Layer: Softmax activation for classification among 10 possible moves
model.add(Dense(10, activation='softmax'))
########### End of CNN Architecture ###########

# Compile the CNN model with an appropriate optimizer and loss function
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary to show details of each layer and number of parameters
model.summary()

# ------------------------------------------------------------------------------
# Train the CNN using the training data extracted from the SUMO simulation.
# Adjust epochs and batch_size based on your specific dataset and training requirements.
model.fit(X_train, Y_train, epochs=10, batch_size=16)

# ------------------------------------------------------------------------------
# Optimal Path Prediction Function:
# After training, we restart the SUMO simulation and use the trained CNN to predict
# an optimal sequence of moves. In this demonstration, the CNN’s prediction at each
# simulation step (using the same feature extraction) is interpreted as the next move.
# The optimal path is printed as a sequence of predicted moves.
def predict_optimal_path(model, simulation_steps=50):
    # Restart the simulation for prediction
    traci.start([sumoBinary, "-c", sumocfg_file])
    optimal_path = []  # To store the sequence of predicted moves
    
    for step in range(simulation_steps):
        # Advance simulation by one step
        traci.simulationStep()
        
        # ---------------------------------------------------------
        # Extract features for the current simulation state.
        # (This should mimic the feature extraction used during training.)
        vehicle_density = np.random.rand(64, 64)
        traffic_lights = np.random.rand(64, 64)
        travel_time = np.random.rand(64, 64)
        
        input_matrix = np.stack([vehicle_density, traffic_lights, travel_time], axis=-1)
        # Expand dimensions since model.predict expects a batch dimension
        input_batch = np.expand_dims(input_matrix, axis=0)
        
        # Get prediction from the trained model
        prediction = model.predict(input_batch)
        move = np.argmax(prediction)  # Choose the move with the highest probability
        
        optimal_path.append(move)
        
        # ---------------------------------------------------------
        # Optional: Add a condition to break early if a destination condition is met.
        # For example, if the simulation indicates that the destination edge has been reached.
        # if destination_reached():
        #     break

    traci.close()
    print("Optimal Path (sequence of predicted moves):", optimal_path)

# ------------------------------------------------------------------------------
# After training, call the function to simulate and display the optimal path
predict_optimal_path(model, simulation_steps=50)
