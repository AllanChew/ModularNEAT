/*
 MIT License

 Copyright (c) 2024 Allan Chew

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
*/

#include "XORTest.h"
#include "./NEAT/NEAT.h"
#include "./NEAT/MathHelpers.h"
#include <iostream>

// solution to XOR (used to calculate fitness)
static const std::vector<std::vector<float>> inputs = { {0,0},{0,1},{1,0},{1,1} };
static const std::vector<float> outputs = { 0,1,1,0 };

XORTest::XORTest(int evals_per_trial_in) : xorNEAT(2, 1, 300, 1.5f), evals_per_trial{ evals_per_trial_in } {}

// helper function for shuffling the contents of the passed in vector
static void Shuffle(std::vector<int>& vec) {
	std::vector<int> inputCopy = vec;
	vec.clear();

	int size = inputCopy.size();

	while (size > 0) {
		int rand_index = NEATMathHelpers::rand_int(size - 1);
		vec.emplace_back(inputCopy[rand_index]);
		if (rand_index != (size - 1)) {
			const int last_elem = inputCopy[size - 1];
			inputCopy[size - 1] = inputCopy[rand_index];
			inputCopy[rand_index] = last_elem;
		}
		--size;
	}
}

NetworkBaseVisual XORTest::Tick() {
	// generate sequence for simulation
	// {0,1,2,3, 0,1,2,3, ...}
	std::vector<int> input_indices;
	input_indices.reserve(evals_per_trial * inputs.size());
	for (int i = 0; i < evals_per_trial; ++i) {
		for (int j = 0; j < inputs.size(); ++j) {
			input_indices.emplace_back(j);
		}
	}

	Shuffle(input_indices); // shuffle sequence to prevent recurrent connections

	// calculate max error (to determine max fitness value)
	const int max_error = 6 * evals_per_trial;
	/*for (auto &e : input_indices) {
		if (outputs[e] == 0) max_error += 1;
		else max_error += 2; // since we're using tanh
	}*/

	// set fitnesses for each organism
	std::vector<float> out = { 0 }; // vector for holding output result
	auto generatedNetworks = xorNEAT.GenerateNetworks();
	xorNEAT.PrintSpecieInfo();
	std::cout << "generation id = " << generation_id << ", numSpecies = " << xorNEAT.GetNumSpecies() << ", numNetworks = " << generatedNetworks.size() << std::endl;
	float max_fitness = 0;
	int max_fitness_index = 0;
	for (int i = 0; i < generatedNetworks.size(); ++i) {
		float fitness = 0;
		for (auto& e : input_indices) {
			std::get<0>(generatedNetworks[i]).Run(inputs[e], out);
			fitness += abs(outputs[e] - out[0]);
		}
		fitness = (max_error - fitness) / max_error;
		std::get<1>(generatedNetworks[i]).SetFitness(fitness);

		// keep track of organism with highest fitness
		if (fitness > max_fitness) {
			max_fitness_index = i;
			max_fitness = fitness;
		}
	}

	std::cout << "max_fitness = " << max_fitness
		<< ", num_nodes = " << std::get<0>(generatedNetworks[max_fitness_index]).GetNumNodes()
		<< ", num_edges = " << std::get<0>(generatedNetworks[max_fitness_index]).GetNumEdges()
		<< ", specie_id = " << std::get<2>(generatedNetworks[max_fitness_index]) << std::endl;

	//std::get<0>(generatedNetworks[max_fitness_index]).Save("xor.dat");
	// print truth table
	std::get<0>(generatedNetworks[max_fitness_index]).Run(inputs[0], out);
	std::cout << "{0,0} => " << out[0] << std::endl;
	std::get<0>(generatedNetworks[max_fitness_index]).Run(inputs[1], out);
	std::cout << "{0,1} => " << out[0] << std::endl;
	std::get<0>(generatedNetworks[max_fitness_index]).Run(inputs[2], out);
	std::cout << "{1,0} => " << out[0] << std::endl;
	std::get<0>(generatedNetworks[max_fitness_index]).Run(inputs[3], out);
	std::cout << "{1,1} => " << out[0] << std::endl;

	xorNEAT.UpdateGeneration();
	++generation_id;

	return std::get<0>(generatedNetworks[max_fitness_index]); // return network of organism with highest fitness for visualization
}
