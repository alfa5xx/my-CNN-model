# Import necessary libraries for TensorFlow/Keras, Stable Baselines3, Gym, and SUMO simulation
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Input,Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Concatenate ,BatchNormalization


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

## -- new changes ----------------------------------------------

from tensorflow.keras.optimizers import Adam, RMSprop

# Option 1: Adam with a lower learning rate
# optimizer = Adam(learning_rate=1e-4)
optimizer = RMSprop(learning_rate=1e-4)


from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from tensorflow.keras.layers import Input

#  new keras model 
# import keras_tuner as kt


#---------------------------------------------------------
def get_vehicle_speed_factor(edge_id, default_max_speed=13.89):
    #
    # Computes a normalized speed factor for a given edge based on the vehicles present.
    # It retrieves all vehicles on the edge, calculates their average speed and then
    # normalizes it by the maximum speed of the edge or a default value.

    # Parameters:
    #     edge_id (str): The ID of the edge.
    #     default_max_speed (float): A fallback max speed (m/s) if no valid max speed can be found.

    # Returns:     float: A speed factor between 0 and 1.
   
    try:
        # Get a list of vehicle IDs currently on the edge
        vehicle_ids = traci.edge.getLastStepVehicleIDs(edge_id)
        if vehicle_ids:
            speeds = []
            for vid in vehicle_ids:
                try:
                    speed = traci.vehicle.getSpeed(vid)
                    speeds.append(speed)
                except Exception as e:
                    print(f"Warning: Could not retrieve speed for vehicle {vid} on edge {edge_id}: {e}")
            if speeds:
                avg_speed = np.mean(speeds)
                # Try to get the edge's max permitted speed from the network.
                try:
                    edge_obj = net.getEdge(edge_id)
                    max_speed = edge_obj.speed if hasattr(edge_obj, 'speed') and edge_obj.speed > 0 else default_max_speed
                except Exception:
                    max_speed = default_max_speed
                # Compute and clip the normalized speed factor between 0 and 1.
                speed_factor = avg_speed / max_speed
                return np.clip(speed_factor, 0, 1)
        # If no vehicles, assume the road is clear and return 1.0.
        return 1.0
    except Exception as e:
        print(f"Error in get_vehicle_speed_factor for edge {edge_id}: {e}")
        return 1.0
#-----------------------------------------------------------------------------------

#-- new   function for keras model tuner ------------------------------------------
# def model_builder(hp):
#     model = Sequential()
#     model.add(BatchNormalization(input_shape=input_shape))
    
#     # Tune the number of filters and the L2 penalty for the first Conv2D layer
#     filters = hp.Choice('filters_1', values=[16, 32, 64])
#     reg = l2(hp.Choice('l2_reg', values=[1e-4, 1e-3, 1e-2]))
    
#     model.add(Conv2D(filters, (3, 3), activation='relu', kernel_regularizer=reg))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
    
#     # You can add more layers similarly
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu', kernel_regularizer=reg))
#     model.add(Dropout(hp.Float('dropout_rate', 0.3, 0.6, step=0.1)))
#     model.add(Dense(n_edges, activation='softmax'))
    
#     # Tune the learning rate for the optimizer
#     hp_lr = hp.Choice('learning_rate', values=[1e-4, 1e-3, 1e-2])
#     model.compile(optimizer=Adam(learning_rate=hp_lr),
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#     return model

# tuner = kt.RandomSearch(model_builder,
#                           objective='val_accuracy',
#                           max_trials=10,
#                           directory='my_dir',
#                           project_name='path_optimization')

# tuner.search(X_train, Y_train, epochs=NUM_EPOCHS, validation_data=(X_test, Y_test))
# best_model = tuner.get_best_models(num_models=1)[0]


#########################################



# ------------------------------------------------------------------------------
# Configuration parameters
TRAIN_EPOCHS = 20            # Number of epochs for training CNN  # default was 150  , 400
TRAIN_SIM_STEPS = 25      # Simulation steps for training data collection# default was 200 , 450 
TEST_SIM_STEPS = 10       # Simulation steps for test data collection # default was 40  , 90
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
# Feature extraction function to get real data from SUMO simulation, with SB3 integration  ---- update for code 52

    
    # Extract features for a given edge from the SUMO simulation.
    # Returns a 3-channel image (64x64 grid) with:
    # - Channel 1: vehicle density (normalized)
    # - Channel 2: traffic light count (normalized)
    # - Channel 3: speed factor (normalized average vehicle speed)
    
def extract_features_modified(edge_id):
    grid_size = (64, 64)
    
    # 1. Vehicle Density (normalized by edge length)
    vehicles = traci.edge.getLastStepVehicleNumber(edge_id)
    edge_length = edge_lengths[edge_id]
    vehicle_density = vehicles / max(edge_length, 1)
    
    # 2. Traffic Light Count (normalized by number of lanes)
    predicted_tls = predict_tls_count_sb3(edge_id)
    edge_obj = net.getEdge(edge_id)
    lanes = [lane.getID() for lane in edge_obj.getLanes()]
    tls_value = predicted_tls / max(len(lanes), 1)
    
    # 3. Speed Factor (aggregated scalar)
    speed_factor = get_vehicle_speed_factor(edge_id)
    
    # Create uniform grids for the image features (2 channels)
    vehicle_grid = np.ones(grid_size) * vehicle_density
    tls_grid = np.ones(grid_size) * tls_value
    # Stack to form an image with 2 channels: channel 1 = vehicle density, channel 2 = traffic lights
    image_features = np.stack([vehicle_grid, tls_grid], axis=-1)
    
    return image_features, speed_factor


# -------------------------------------------------------------------------------
# Multi-Input Model Definition:
#
# This function builds a CNN that accepts two inputs:
#   - 'image_input' with shape (64, 64, 2): the spatial features
#   - 'scalar_input' with shape (1,): the aggregated speed factor
def build_multi_input_model(image_input_shape, scalar_input_shape=(1,), n_edges=10):
    # -------------------- Image Branch --------------------
    image_input = Input(shape=image_input_shape, name='image_input')
    x = BatchNormalization()(image_input)
    x = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    
    # -------------------- Scalar Branch (Speed Factor) --------------------
    scalar_input = Input(shape=scalar_input_shape, name='scalar_input')
    y = Dense(16, activation='relu')(scalar_input)
    
    # -------------------- Merge Branches --------------------
    combined = Concatenate()([x, y])
    combined = Dense(128, activation='relu')(combined)
    output = Dense(n_edges, activation='softmax')(combined)
    
    model = Model(inputs=[image_input, scalar_input], outputs=output)
    return model


# ------------------------------------------------------------------------------
# Path evaluation function combining multiple criteria
def evaluate_path(path):
   
    # Evaluate a path based on:
    #   - Total distance (edge lengths)
    #   - Total number of vehicles (waiting time indicator)
    #   - Total traffic lights count
    #   - Number of edges (path length)
    # Lower score is better.
    
    try:
        if not path:
            return float('inf')
        total_length = sum(edge_lengths[edge] for edge in path)
        total_vehicles = sum(traci.edge.getLastStepVehicleNumber(edge) for edge in path)
        total_tls = sum(predict_tls_count_sb3(edge) for edge in path)
        # The number of edges directly represents the path length
        num_edges = len(path)
        # Enforce traffic light limitation
        if total_tls > 2:
           return float('inf')
       
        # --- New Code: Compute speed penalty -----------------------------------------------------------
        speed_factors = []
        for edge in path:
            try:
                edge_obj = net.getEdge(edge)
                current_speed = traci.edge.getLastStepMeanSpeed(edge)
                max_speed = edge_obj.speed
                if max_speed > 0:
                    speed_factors.append(current_speed / max_speed)
            except Exception as e:
                print(f"Error calculating speed for edge {edge}: {e}")
        # Average speed factor for the path; if no values then assume full speed (1)
        avg_speed_factor = np.mean(speed_factors) if speed_factors else 1
        # Define desired minimum speed factor for an uncongested edge.
        desired_speed = 0.6   # adjust this threshold as needed
        speed_penalty = 0
        if avg_speed_factor < desired_speed:
            # The penalty weight can be tuned (here, 100 is used as an example)
            speed_penalty = (desired_speed - avg_speed_factor) * 100
#------------------------------------------------------------------------------------- penelize for low speed in each edge
        
        # Weight factors (can be adjusted based on simulation characteristics)
        length_weight = 0.3
        vehicle_weight = 0.3
        tls_weight = 0.2
        edge_count_weight = 0.2
        
        score = (length_weight * total_length +
                 vehicle_weight * total_vehicles +
                 tls_weight * total_tls +
                 edge_count_weight * num_edges+
                 speed_penalty)
        return score
    except Exception as e:
        print(f"Error evaluating path: {e}")
        return float('inf')

# ------------------------------------------------------------------------------
# Function to generate all possible loop-free paths (DFS with max_depth limit)
def generate_all_paths(start_edge, destination_edge, max_depth=20):
    
    # Generate all possible loop-free paths from start_edge to destination_edge.
    
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
def get_speed_range_for_path(path):
    
    # Calculates the minimum and maximum mean speeds along a given optimal path.
    
    # The function iterates over each edge in the path and uses traci.edge.getLastStepMeanSpeed(edge)
    # to retrieve the current mean speed of that edge. It then returns the minimum and maximum speed values.
    
    # Parameters:    #     path (list of str): List of edge IDs along the optimal path.
        
    # Returns:  #     (min_speed, max_speed): Tuple with minimum and maximum mean speeds for the path.
    #    Returns (None, None) if no speeds are recorded.
   
    speeds = []
    for edge in path:
        try:
            # Retrieve the mean speed for the edge using SUMO's traci API.
            mean_speed = traci.edge.getLastStepMeanSpeed(edge)
            speeds.append(mean_speed)
        except Exception as e:
            print(f"Error retrieving speed for edge {edge}: {e}")
    if speeds:
        return min(speeds), max(speeds)
    else:
        return None, None



# ------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# Training Data Collection (Modified):
#
# In this function, we use the modified feature extraction function to collect
# separate inputs for the image features and the speed factor scalar.
def collect_training_data_modified(start_edge, destination_edge, simulation_steps=25):
    X_train_image = []
    X_train_scalar = []
    Y_train = []
    
    traci.start(sumo_cmd)
    print("SUMO started for training data collection with modified features")
    
    all_paths = generate_all_paths(start_edge, destination_edge)
    if not all_paths:
        print("No valid paths found. Check connectivity.")
        traci.close()
        return np.array([]), np.array([]), np.array([])
    
    # Sample at a chosen interval (e.g., every 5 simulation steps)
    for step in range(simulation_steps):
        traci.simulationStep()
        if step % 5 != 0:
            continue
        print(f"Modified Training simulation step {step}/{simulation_steps}")
        path_scores = [(path, evaluate_path(path)) for path in all_paths]
        path_scores.sort(key=lambda x: x[1])
        best_paths = [p for p, _ in path_scores[:min(5, len(path_scores))]]
        
        # Record the best paths generated in this simulation step
        record_training_routes(best_paths)
        
        for path in best_paths:
            for i in range(len(path)-1):
                current_edge = path[i]
                next_edge = path[i+1]
                image_feat, speed_val = extract_features_modified(current_edge)
                X_train_image.append(image_feat)
                X_train_scalar.append([speed_val])  # Keep as a list to ensure the shape is (1,)
                label = np.zeros(len(edge_to_index))
                label[edge_to_index[next_edge]] = 1
                Y_train.append(label)
    print(f"Collected {len(X_train_image)} modified training samples")
    traci.close()
    return np.array(X_train_image), np.array(X_train_scalar), np.array(Y_train)

# -------------------------------------------------------------------------------
# Training Function for the Multi-Input Model:
def train_multi_input_model(X_train_image, X_train_scalar, Y_train, 
                            X_test_image, X_test_scalar, Y_test, n_edges):
    model = build_multi_input_model((64, 64, 2), (1,), n_edges)
    model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    early_stop = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True) # default  patience=20
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10) # default value for factor=0.6, patience=5
    
    history = model.fit(
        [X_train_image, X_train_scalar],
        Y_train,
        validation_data=([X_test_image, X_test_scalar], Y_test),
        epochs=TRAIN_EPOCHS,
        batch_size=64,
        callbacks=[early_stop, reduce_lr]
    )
    
    # Extract start and end points for each metric
    train_acc_start = history.history['accuracy'][0]
    train_acc_end = history.history['accuracy'][-1]
    val_acc_start = history.history['val_accuracy'][0]
    val_acc_end = history.history['val_accuracy'][-1]
    train_loss_start = history.history['loss'][0]
    train_loss_end = history.history['loss'][-1]
    val_loss_start = history.history['val_loss'][0]
    val_loss_end = history.history['val_loss'][-1]

    # Print the values to console
    print("\n--- Training Metrics Start and End Points ---")
    print(f"Train Accuracy: {train_acc_start:.4f} → {train_acc_end:.4f} (Change: {((train_acc_end - train_acc_start) / train_acc_start * 100):.2f}%)")
    print(f"Validation Accuracy: {val_acc_start:.4f} → {val_acc_end:.4f} (Change: {((val_acc_end - val_acc_start) / val_acc_start * 100):.2f}%)")
    print(f"Train Loss: {train_loss_start:.4f} → {train_loss_end:.4f} (Change: {((train_loss_end - train_loss_start) / train_loss_start * 100):.2f}%)")
    print(f"Validation Loss: {val_loss_start:.4f} → {val_loss_end:.4f} (Change: {((val_loss_end - val_loss_start) / val_loss_start * 100):.2f}%)")

    # Create the figure
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

    # Mark the start and end points
    plt.scatter(0, train_acc_start, color='blue', s=50, zorder=5)
    plt.scatter(len(history.history['accuracy'])-1, train_acc_end, color='blue', s=50, zorder=5)
    plt.scatter(0, val_acc_start, color='orange', s=50, zorder=5)
    plt.scatter(len(history.history['val_accuracy'])-1, val_acc_end, color='orange', s=50, zorder=5)

    # Add text annotations for the values
    plt.annotate(f"{train_acc_start:.4f}", (0, train_acc_start), textcoords="offset points", 
                xytext=(0,10), ha='center')
    plt.annotate(f"{train_acc_end:.4f}", (len(history.history['accuracy'])-1, train_acc_end), 
                textcoords="offset points", xytext=(0,10), ha='center')
    plt.annotate(f"{val_acc_start:.4f}", (0, val_acc_start), textcoords="offset points", 
                xytext=(0,-15), ha='center')
    plt.annotate(f"{val_acc_end:.4f}", (len(history.history['val_accuracy'])-1, val_acc_end), 
                textcoords="offset points", xytext=(0,-15), ha='center')

    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')

    # Mark the start and end points
    plt.scatter(0, train_loss_start, color='blue', s=50, zorder=5)
    plt.scatter(len(history.history['loss'])-1, train_loss_end, color='blue', s=50, zorder=5)
    plt.scatter(0, val_loss_start, color='orange', s=50, zorder=5)
    plt.scatter(len(history.history['val_loss'])-1, val_loss_end, color='orange', s=50, zorder=5)

    # Add text annotations for the values
    plt.annotate(f"{train_loss_start:.4f}", (0, train_loss_start), textcoords="offset points", 
                xytext=(0,10), ha='center')
    plt.annotate(f"{train_loss_end:.4f}", (len(history.history['loss'])-1, train_loss_end), 
                textcoords="offset points", xytext=(0,10), ha='center')
    plt.annotate(f"{val_loss_start:.4f}", (0, val_loss_start), textcoords="offset points", 
                xytext=(0,-15), ha='center')
    plt.annotate(f"{val_loss_end:.4f}", (len(history.history['val_loss'])-1, val_loss_end), 
                textcoords="offset points", xytext=(0,-15), ha='center')

    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    print("\n")
    print("\n")
    print("\n")
    # Add training parameters as text annotation
    epoch_info = f"epoch {TRAIN_EPOCHS}\ntrain sim {TRAIN_SIM_STEPS}\ntest sim {TEST_SIM_STEPS}\nearly stop {early_stop.patience}\nreduce lr factor {reduce_lr.factor}\npatience {reduce_lr.patience}"
    plt.figtext(0.95, 0.7, epoch_info, fontsize=10, ha='right')
    
    # Add training metrics summary
    metrics_text = (
        f"Train Acc: {train_acc_start:.2f} → {train_acc_end:.2f}\n"
        f"Val Acc: {val_acc_start:.2f} → {val_acc_end:.2f}\n"
        f"Train Loss: {train_loss_start:.2f} → {train_loss_end:.2f}\n"
        f"Val Loss: {val_loss_start:.2f} → {val_loss_end:.2f}"
        f"SUMO starte and end edges: paths from {start_edge} to {destination_edge}"
    )
    plt.figtext(0.01, 0.7, metrics_text, fontsize=10, ha='left', va='top')

    plt.tight_layout()
    plt.savefig('training_history.png')
    
    
    # Save metrics to a separate file for future reference
    with open('training_metrics.txt', 'w') as f:
        f.write("--- Training Metrics Start and End Points ---\n")
        f.write(f"Train Accuracy: {train_acc_start:.4f} → {train_acc_end:.4f} (Change: {((train_acc_end - train_acc_start) / train_acc_start * 100):.2f}%)\n")
        f.write(f"Validation Accuracy: {val_acc_start:.4f} → {val_acc_end:.4f} (Change: {((val_acc_end - val_acc_start) / val_acc_start * 100):.2f}%)\n")
        f.write(f"Train Loss: {train_loss_start:.4f} → {train_loss_end:.4f} (Change: {((train_loss_end - train_loss_start) / train_loss_start * 100):.2f}%)\n")
        f.write(f"Validation Loss: {val_loss_start:.4f} → {val_loss_end:.4f} (Change: {((val_loss_end - val_loss_start) / val_loss_start * 100):.2f}%)\n")
    
    if SHOW_PLOT:
        plt.show()
    else:
        plt.close()
    
    return model


#--------------------------------------------------------------------------------------

def collect_test_data_modified(start_edge, destination_edge, simulation_steps=TEST_SIM_STEPS):
    """
    Collect modified test data with separate outputs for image features,
    scalar speed factor, and one-hot encoded labels.
    """
    X_test_image = []
    X_test_scalar = []
    Y_test = []
    try:
        traci.start(sumo_cmd)
        print("SUMO started for test data collection with modified features")
        all_paths = generate_all_paths(start_edge, destination_edge)
        if not all_paths:
            print("No valid paths found for testing.")
            traci.close()
            return np.array([]), np.array([]), np.array([])
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
                    # Use the modified feature extraction: returns image features and scalar speed value.
                    image_feat, scalar_feat = extract_features_modified(current_edge)
                    X_test_image.append(image_feat)
                    X_test_scalar.append([scalar_feat])  # Ensuring it's the correct shape.
                    # Create one-hot label for the next edge.
                    label = np.zeros(len(edge_to_index))
                    label[edge_to_index[next_edge]] = 1
                    Y_test.append(label)
        print(f"Collected {len(X_test_image)} modified test samples")
        traci.close()
        return np.array(X_test_image), np.array(X_test_scalar), np.array(Y_test)
    except Exception as e:
        print(f"Error collecting test data: {e}")
        if traci.isLoaded():
            traci.close()
        return np.array([]), np.array([]), np.array([])


# ------------------------------------------------------------------------------
# Define CNN model architectures with BatchNormalization to help training

# Example for the original model architecture modification:
def build_original_model(input_shape):
    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(1e-4)))
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
# Plot training history
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
    
    model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    early_stop = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True) # default  patience=20
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10) # default value for factor=0.6, patience=5
    
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=TRAIN_EPOCHS,
        batch_size=64,
        callbacks=[early_stop, reduce_lr]
    )
    
    # Extract start and end points for each metric
    train_acc_start = history.history['accuracy'][0]
    train_acc_end = history.history['accuracy'][-1]
    val_acc_start = history.history['val_accuracy'][0]
    val_acc_end = history.history['val_accuracy'][-1]
    train_loss_start = history.history['loss'][0]
    train_loss_end = history.history['loss'][-1]
    val_loss_start = history.history['val_loss'][0]
    val_loss_end = history.history['val_loss'][-1]

    # Print the values to console
    print("\n--- Training Metrics Start and End Points ---")
    print(f"Train Accuracy: {train_acc_start:.4f} → {train_acc_end:.4f} (Change: {((train_acc_end - train_acc_start) / train_acc_start * 100):.2f}%)")
    print(f"Validation Accuracy: {val_acc_start:.4f} → {val_acc_end:.4f} (Change: {((val_acc_end - val_acc_start) / val_acc_start * 100):.2f}%)")
    print(f"Train Loss: {train_loss_start:.4f} → {train_loss_end:.4f} (Change: {((train_loss_end - train_loss_start) / train_loss_start * 100):.2f}%)")
    print(f"Validation Loss: {val_loss_start:.4f} → {val_loss_end:.4f} (Change: {((val_loss_end - val_loss_start) / val_loss_start * 100):.2f}%)")

    # Create the figure
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

    # Mark the start and end points
    plt.scatter(0, train_acc_start, color='blue', s=50, zorder=5)
    plt.scatter(len(history.history['accuracy'])-1, train_acc_end, color='blue', s=50, zorder=5)
    plt.scatter(0, val_acc_start, color='orange', s=50, zorder=5)
    plt.scatter(len(history.history['val_accuracy'])-1, val_acc_end, color='orange', s=50, zorder=5)

    # Add text annotations for the values
    plt.annotate(f"{train_acc_start:.4f}", (0, train_acc_start), textcoords="offset points", 
                xytext=(0,10), ha='center')
    plt.annotate(f"{train_acc_end:.4f}", (len(history.history['accuracy'])-1, train_acc_end), 
                textcoords="offset points", xytext=(0,10), ha='center')
    plt.annotate(f"{val_acc_start:.4f}", (0, val_acc_start), textcoords="offset points", 
                xytext=(0,-15), ha='center')
    plt.annotate(f"{val_acc_end:.4f}", (len(history.history['val_accuracy'])-1, val_acc_end), 
                textcoords="offset points", xytext=(0,-15), ha='center')

    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')

    # Mark the start and end points
    plt.scatter(0, train_loss_start, color='blue', s=50, zorder=5)
    plt.scatter(len(history.history['loss'])-1, train_loss_end, color='blue', s=50, zorder=5)
    plt.scatter(0, val_loss_start, color='orange', s=50, zorder=5)
    plt.scatter(len(history.history['val_loss'])-1, val_loss_end, color='orange', s=50, zorder=5)

    # Add text annotations for the values
    plt.annotate(f"{train_loss_start:.4f}", (0, train_loss_start), textcoords="offset points", 
                xytext=(0,10), ha='center')
    plt.annotate(f"{train_loss_end:.4f}", (len(history.history['loss'])-1, train_loss_end), 
                textcoords="offset points", xytext=(0,10), ha='center')
    plt.annotate(f"{val_loss_start:.4f}", (0, val_loss_start), textcoords="offset points", 
                xytext=(0,-15), ha='center')
    plt.annotate(f"{val_loss_end:.4f}", (len(history.history['val_loss'])-1, val_loss_end), 
                textcoords="offset points", xytext=(0,-15), ha='center')

    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Add training parameters as text annotation
    epoch_info = f"epoch {TRAIN_EPOCHS}\ntrain sim {TRAIN_SIM_STEPS}\ntest sim {TEST_SIM_STEPS}\nearly stop {early_stop.patience}\nreduce lr factor {reduce_lr.factor}\npatience {reduce_lr.patience}"
    plt.figtext(0.95, 0.7, epoch_info, fontsize=10, ha='right')

    plt.tight_layout()
    plt.savefig('training_history.png')
    
    # Save metrics to a separate file for future reference
    with open('training_metrics.txt', 'w') as f:
        f.write("--- Training Metrics Start and End Points ---\n")
        f.write(f"Train Accuracy: {train_acc_start:.4f} → {train_acc_end:.4f} (Change: {((train_acc_end - train_acc_start) / train_acc_start * 100):.2f}%)\n")
        f.write(f"Validation Accuracy: {val_acc_start:.4f} → {val_acc_end:.4f} (Change: {((val_acc_end - val_acc_start) / val_acc_start * 100):.2f}%)\n")
        f.write(f"Train Loss: {train_loss_start:.4f} → {train_loss_end:.4f} (Change: {((train_loss_end - train_loss_start) / train_loss_start * 100):.2f}%)\n")
        f.write(f"Validation Loss: {val_loss_start:.4f} → {val_loss_end:.4f} (Change: {((val_loss_end - val_loss_start) / val_loss_start * 100):.2f}%)\n")
    
    training_metrics_summary = (
    f"--- Training Metrics Start and End Points ---\n"
    f"Train Accuracy: {train_acc_start:.4f} → {train_acc_end:.4f} (Change: {((train_acc_end - train_acc_start) / train_acc_start * 100):.2f}%)\n"
    f"Validation Accuracy: {val_acc_start:.4f} → {val_acc_end:.4f} (Change: {((val_acc_end - val_acc_start) / val_acc_start * 100):.2f}%)\n"
    f"Train Loss: {train_loss_start:.4f} → {train_loss_end:.4f} (Change: {((train_loss_end - train_loss_start) / train_loss_start * 100):.2f}%)\n"
    f"Validation Loss: {val_loss_start:.4f} → {val_loss_end:.4f} (Change: {((val_loss_end - val_loss_start) / val_loss_start * 100):.2f}%)"
)

    if SHOW_PLOT:
        plt.show()
    else:
        plt.close()
    
    return model

# ------------------------------------------------------------------------------
# New function: Find the optimal path from start_edge to destination_edge
def find_optimal_path_all(model, start_edge, destination_edge):
   
    # Generate all possible paths from start_edge to destination_edge (loop-free) and select the optimal path.
    # The optimal path minimizes a composite cost:
    #   - Total distance (edge lengths)
    #   - Total vehicles on the path
    #   - Total traffic lights along the path
    #   - Number of edges in the path
   
    try:
        traci.start(sumo_cmd)
        print("SUMO started for optimal path search")
        all_paths = generate_all_paths(start_edge, destination_edge, max_depth=50)
        valid_paths = [path for path in all_paths if count_traffic_lights_in_path(path) <= 2]
        if not all_paths:
            print("No valid paths found from start to destination.")
            traci.close()
            return []
        # Evaluate each path and choose the one with minimum cost
        optimal_path = min(valid_paths, key=lambda path: evaluate_path(path))
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
  
    # Count the total number of traffic lights along a given path.
    
    # For each edge in the path, the function checks whether the destination node (junction)
    # is controlled by a traffic light. It uses the node type from the SUMO network, and if 
    # the simulation is running (via traci), it also verifies using traci.trafficlight.getIDList().
    
    # Parameters:
    #     path (list): A list of edge IDs representing the route.
    
    # Returns:
    #     int: Total count of traffic lights in the path.
   
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

#------------------------------------ new code from 43 trained data usage ------------------------------------------------------------------------

def find_path_between_edges(training_routes, start_edge, end_edge):
    
    # Find and extract the sub-path between the specified start and end edges 
    # from a list of training routes.

    # Args:
    #     training_routes (List[List[str]]): List of previously generated routes.
    #     start_edge (str): Starting edge ID.
    #     end_edge (str): Destination edge ID.

    # Returns:
    #     List[str]: The sub-path from start_edge to end_edge (inclusive), or None if not found.

    # Iterate over training routes looking for one that contains both edges
    for route in training_routes:
        if start_edge in route and end_edge in route:
            start_idx = route.index(start_edge)
            end_idx = route.index(end_edge)
            if start_idx < end_idx:
                # Return the segment of the route between start and end (inclusive)
                return route[start_idx:end_idx + 1]
    return None

#-----------------  new code from code 44 -----------

def random_edge_path_selection(net, training_routes):

    # Interactively select random edges and find a valid sub-path from training routes.

    # Args:
    #     net (sumolib.net.Net): SUMO network object.
    #     training_routes (List[List[str]]): List of previously generated routes.

    # Returns:
    #     List[str]: Selected sub-path from start_edge to end_edge, or None.
   
    # Collect all unique edges from training routes
    unique_edges = set()
    for route in training_routes:
        unique_edges.update(route)
    
    # Sort edges for easier reading
    sorted_unique_edges = sorted(list(unique_edges))
    
    # Print all unique edges
    print("\n--- Edges Used in Training Routes ---")
    print("Total unique edges:", len(sorted_unique_edges))
    print("Available Edges:")
    for edge in sorted_unique_edges:
        print(edge)
    
    while True:
        try:
            # Input for start edge directly by edge ID
            start_edge = input("\nEnter the START edge ID (exactly as shown above): ").strip()
            
            # Input for destination edge directly by edge ID
            end_edge = input("Enter the END edge ID (exactly as shown above): ").strip()
            
            # Validate input: start and end must be different and must exist in unique edges
            if start_edge not in unique_edges or end_edge not in unique_edges:
                print("Invalid edge ID. Please use exact edge IDs from the list above.")
                continue
            
            if start_edge == end_edge:
                print("Start and destination edges must be different!")
                continue
            
            # Find the sub-path between the selected edges
            selected_path = find_path_between_edges(training_routes, start_edge, end_edge)
            
            if selected_path:
                print("\nFound Path:")
                print(" -> ".join(selected_path))
                
                # Start SUMO simulation to get real-time metrics
                try:
                    traci.start(sumo_cmd)
                    
                    # Calculate route metrics
                    route_length = sum(edge_lengths.get(edge, 0) for edge in selected_path)
                    vehicle_count = sum(traci.edge.getLastStepVehicleNumber(edge) for edge in selected_path)
                    traffic_lights = count_traffic_lights_in_path(selected_path)
                    
                    print("\nRoute Metrics:")
                    print(f"Length: {route_length:.2f} meters")
                    print(f"Total Vehicles: {vehicle_count}")
                    print(f"Traffic Lights: {traffic_lights}")
                    
                    traci.close()  # Close SUMO simulation
                    
                    return selected_path  # Valid path found; exit function.
                
                except Exception as e:
                    print(f"Error during SUMO simulation for route metrics: {e}")
                    if traci.isLoaded():
                        traci.close()
                    return None
            
            else:
                print("No valid sub-path found between selected edges in training routes.")
                retry = input("Do you want to try again? (y/n): ").lower()
                if retry != 'y':
                    break
                
        except Exception as e:
            print(f"An error occurred: {e}")
            retry = input("Do you want to try again? (y/n): ").lower()
            if retry != 'y':
                break
    
    return None



# ------------------------------------------------------------------------------
# SUMO command setup (use "sumo-gui" for visualization if desired)
sumoBinary = "sumo"
sumo_cmd = [sumoBinary, "-c", sumocfg_file]

# ------------------------------------------------------------------------------
# Main execution block
if __name__ == "__main__":
    start_edge = "E0"        # Replace with your actual start edge ID
    destination_edge = "E3"  # Replace with your actual destination edge ID
    # n_edges = len(edge_to_index)  # Number of possible next edges
    
    # Collect modified training and test data
    X_train_image, X_train_scalar, Y_train = collect_training_data_modified(start_edge, destination_edge, simulation_steps=TRAIN_SIM_STEPS)
    X_test_image, X_test_scalar, Y_test = collect_test_data_modified(start_edge, destination_edge, simulation_steps=TEST_SIM_STEPS)

    
    # Train the CNN model if data is available
    if X_train_image.size and Y_train.size:
        cnn_model = train_cnn_model(X_train_image, Y_train, X_test_image, Y_test)
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
        
        # Get the speed range for the optimal path:
        min_speed, max_speed = get_speed_range_for_path(optimal_path)
        if min_speed is not None and max_speed is not None:
            print("Speed range for optimal path: {:.2f} m/s - {:.2f} m/s".format(min_speed, max_speed))
        else:
            print("Speed information not available for the optimal path.")
        
        
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