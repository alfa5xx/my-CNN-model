There are two type of code. In the first try , we  have attempted to create a python code in order to find out an optimal path based on number of vehicles, number of traffic lights, number of edges,  minimum travel time and destination in the optimal path. The codes are located in folder "Without speed factor".  Then, we have attmepted to improve the proposed code to find out an optimal path in a manner to use Speed factor as the second factor in the CNN. In this way, the proposed CNN benefits from the mentioend prior factors as the first input and the speed factor as the second input  within the proposed CNN.

The most important part is that.. we have to install Eclipse SUMO - Simulation of Urban MObility somulation which is a 2D simulation for simulating the map and edges and routes on a map.

next we have to benefit from a Python IDE , in this case, we used Spyder which is installed on anaconda package as the Python IDE.

In the b.sumocfg  , there is a section that is related to required files location such as  b.sumocfg. b.net.xml and b.rou.xml .

after set these files location in the b.sumocfg file we are able t run the simulation.

-
