This model was built, which is connected to research in the title "Convolutional Neural Networks with Stable-Baselines for Optimized Path Planning by Simulation".

There are two types of codes. In the first try , we have attempted to create a model in Python code in order to find an optimal path based on the number of vehicles, number of traffic lights, number of edges, minimum travel time, and destination in the optimal path. The codes are located in the folder "Without speed factor". Then, we have attempted to improve the proposed code to find an optimal path in a manner that uses the Speed factor as the second factor in the CNN. In this way, the proposed CNN benefits from the mentioned prior factors as the first input and the speed factor as the second input within the proposed CNN.

The most important part is that.. -Install requirement--- We must install Python language 3.13 and Simulation of Urban MObility somulation -SUMO- which is a 2D simulation for simulating the map, edges, and routes on a map.

---- Procedure-- Next , we have to benefit from a Python IDE; in this case, we have used Spyder IDE v5.7 , which is installed on Anaconda Package-Anaconda Distribution 2025.06 in January 2025 - as the Python IDE.

--- Configuration files before running a simulation --

* All requirement files are b.sumocfg. b.net.xml and b.rou.xml , must be in the same folder.
* Edit the b.sumocfg; there is a section that is related to the required files' location, such as b.sumocfg. b.net.xml and b.rou.xml. It must be set correctly.
* Once the file locations are set, configure hyperparameters like epochs and learning rate as needed in the code. Then run the simulation in Spyder IDE.
* Note: In some cases, we need to clear the cache and re-run the simulation.
* At the end, an optimal route and figure will be displayed in the simulation outcome.

The Python codes, which were labeled 'Ok-Ok-OK' in their names, work well. Also, others that have different names work well, but a stable version of code was written in the code, which was marked with Ok-Ok-Ok in their names.

Good luck !
