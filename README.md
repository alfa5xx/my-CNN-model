This model was built which is connected to a research in this title "Convolutional Neural Networks with Stable-Baselines for Optimized Path Planning by Simulation".

There are two type of codes. In the first try , we  have attempted to create a model in  python code in order to find out an optimal path based on number of vehicles, number of traffic lights, number of edges,  minimum travel time and destination in the optimal path. The codes are located in folder "Without speed factor".  Then, we have attmepted to improve the proposed code to find out an optimal path in a manner to use Speed factor as the second factor in the CNN. In this way, the proposed CNN benefits from the mentioend prior factors as the first input and the speed factor as the second input  within the proposed CNN.

The most important part is that.. 
-Install requirement---
We  must install Python language and Simulation of Urban MObility somulation -SUMO- which is a 2D simulation for simulating the map and edges and routes on a map.

---- Procedure--
Next , we have to benefit from a Python IDE , in this case, we haveused Spyder IDE which is installed on Anaconda Package as the Python IDE.


--- Configuration files before run a simulation -- 

- All requirement files are b.sumocfg. b.net.xml and b.rou.xml, they must be  in  as same folder.

- Edit the b.sumocfg  , there is a section that is related to required files' location such as  b.sumocfg. b.net.xml and b.rou.xml. It must be set correctly.

- After set the files location, we are able to find hyper parameters configuration in the codes such as: number of epoches, learning rate and etc in the code context . Modify them based on your simulation need and run the simulation on Spyder IDE.

- Note: In some case, We need to clear cache and re-run the simulation.


- At the end, an optimal route and figure will be displayed in the simulation outcome.

The last Python code that was labaled Ok-Ok-OK in its name , they work well . Also, others which have different name work well but stable version of code was written in the codde whcih was marked with Ok-Ok-Ok in their names.

Good luck !
