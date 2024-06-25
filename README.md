## Background

NeuroEvolution of Augmenting Topologies (NEAT) is a genetic algorithm (GA) for the generation of evolving artificial neural networks (a neuroevolution technique) developed by Kenneth Stanley and Risto Miikkulainen in 2002 while at The University of Texas at Austin. [Source: [Wikipedia](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies)]

You can learn more about NEAT on the [NEAT website](https://nn.cs.utexas.edu/?neat) and/or by reading through the 3 academic papers: [journal paper](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf), [conference paper](https://nn.cs.utexas.edu/downloads/papers/stanley.gecco02_1.pdf), [shorter conference paper](https://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf).

<center><img src="/Images/xor.gif" width = "700"></center>

Contained in this repo is a lightweight and easy-to-use C++ module for implementing NEAT into your own C++ projects. The module is located inside the *NEAT* folder and has been coded in C++ using only the C++ Standard Library; and so no external dependencies are required.

The code in this repo has been tested and compiled using Visual C++ (2019) and is licensed under the **[MIT License](LICENSE)**.

## Visualization (optional)

*[NEATViz.cpp](Source/NEATViz.cpp)* contains basic code for how to visualize the neural networks. You can alternatively implement your own visualization tool (in which case you can use *NEATViz.cpp* as a guide). Or you can opt to not use any visualization at all. The choice is up to you.

*NEATViz.cpp* uses the `NetworkBaseVisual` class to determine how to draw the neural networks. `NetworkBaseVisual` inherits from `NetworkBase` and contains extra visualization information, but this also means that it uses more memory. So if you won't be using the visualization information (i.e. only running inference), then you can improve performance by only using `NetworkBase` instead.

To compile *NEATViz.cpp* from source, you'll need to import [SDL](https://github.com/libsdl-org/SDL) and the [SDL_tff extension library](https://github.com/libsdl-org/SDL_ttf). If you're new to SDL, there are [several resources](https://wiki.libsdl.org/SDL2/Tutorials) you can use to get started with SDL.

*NEATViz.cpp* was tested using [SDL2 (v2.30.3)](https://github.com/libsdl-org/SDL/releases/tag/release-2.30.3) and [SDL2_tff (v2.22.0)](https://github.com/libsdl-org/SDL_ttf/releases/tag/release-2.22.0). A pre-compiled binary of *NEATViz.cpp* is also available [here](https://github.com/AllanChew/ModularNEAT/releases).

### Controls

After running *NEATViz.cpp*, you can use the below controls to step through the XOR test example:

**Right Arrow Key** - Step forward to the next generation\
**Mouse Wheel** - Adjust vertical alignment of the rendered neurons

The console and window will update as you step through each generation. The console contains useful information about each generation (e.g. number of species, max fitness, etc.), and the window contains the visualization of the best network (highest fitness) from the current generation.

<center><img src="/Images/network.png" width = "600"></center>

Forward connections are colored based on their weight values; with positive values being green, negative values being red, and values close to 0 being black.
Recurrent connections are always colored blue and drawn with a curve.
In the network above, nodes 0 to 3 are input nodes (with node 3 being the bias node), nodes 4 and 5 are output nodes, and a recurrent connection exists from node 4 to node 8.

## Setup and Usage

To import the module into your own C++ project, copy and paste the *NEAT* folder into your project's source directory.

In order to use the module, you'll need to `#include` *NEAT/NEAT.h* and/or *NEAT/Network.h* into your own source code (keeping in mind the locations of the aforementioned files).
And you'll also need to compile and link the *.cpp* files into your final binary.

For an example of how to use the module's interface, you can look at *XORTest.cpp/h*. In summary, you'll need to do the following steps:

1. Instantiate `NEAT` with the desired parameters (e.g. network input and output size, population size, mutation rates, etc.)
2. Call `NEAT::GenerateNetworks` to create the neural networks for the current generation
3. Evaluate the networks using `NetworkBase::Run`
4. Score the networks using `FitnessInterface::SetFitness`
5. Call `NEAT::UpdateGeneration` to update the generation
6. Repeat steps 2-5 until some termination condition (e.g. fitness reaches some desired value)

Once you find a network you like, you can save it to a file using `NetworkBaseVisual::Save`.
To load a network that's been saved to a file, use the `NetworkBase` and/or `NetworkBaseVisual` constructor(s) with the name/path of the file as the argument. The code below shows how to load and run a saved network.

```c
NetworkBase network("xor.dat");

// run the network on some data
std::vector<float> out = {0};
network.Run<bool>({false,false}, out);
std::cout << "{0,0} => " << out[0] << std::endl;
network.Run<bool>({false,true}, out);
std::cout << "{0,1} => " << out[0] << std::endl;
network.Run<bool>({true,false}, out);
std::cout << "{1,0} => " << out[0] << std::endl;
network.Run<bool>({true,true}, out);
std::cout << "{1,1} => " << out[0] << std::endl;
```

## Cloning Networks

If you're going to be using the same neural network in multiple places simultaneously, then you should use the copy constructor/assignment to create additional clones of the network, instead of just loading the same network again from the same file.

This will ensure that shared information between the networks (e.g. weight information) isn't duplicated. Below is an example for how to create clones of a network.

```c
NetworkBase enemyA_network1("enemyA.dat"); // network loaded from file

NetworkBase enemyA_network2 = enemyA_network1; // network cloned using copy ctor

NetworkBase enemyA_network3;
enemyA_network3 = enemyA_network1; // network cloned using copy assignmnent
```

In the example above, you might be wondering why you would ever want to create duplicate clones of a neural network (e.g. why not use a single instance instead?).

There are 2 possible reasons:
1. One reason is that the neural network could contain recurrent connections. With recurrent neural networks, networks have a hidden state that depends on the sequence of data passed into the network. If a single network is used, data from different agents would get mixed together and corrupt the output of these recurrent connections; therefore making the entire output of the network invalid.
2. Another reason is that it allows the agents to run their networks in parallel. As an aside, the shared data inside the networks is read-only, and so running these networks in parallel is perfectly valid and won't cause any race conditions.
