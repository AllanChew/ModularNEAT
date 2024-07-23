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

#pragma once

#include <map>
#include <memory>
#include "Genome.h"

// interface to set the fitness of an organism
// does a safety check to ensure the organism is still alive
// shouldn't need to initialize this directly; gets instantiated through NEAT::GenerateNetworks
class FitnessInterface {
public:
	FitnessInterface(const std::shared_ptr<int>& fitness_valid_ptr_in, float& fitness_ref_in);
	bool SetFitness(float f);

private:
	std::weak_ptr<int> fitness_valid_ptr;
	float& fitness_ref;
};

class NEAT {
public:
	// input size, output size, and population size should all be greater than 0
	NEAT(int input_size = 1, int output_size = 1, int pop_size_in = 150, float compatibility_thresh_in = 1.5f, float c1_c2_in = 1.f, float c3_in = 0.4f, float top_p_cutoff_in = 0.6, float add_node_mutation_prob_in = 0.03f, float add_edge_mutation_prob_in = 0.3f, float weight_mutation_prob_in = 0.8);

	bool Load(const char* fname); // returns false if it fails to open file
	void Save(const char* fname) const;

	std::vector<std::tuple<NetworkBaseVisual, FitnessInterface, int>> GenerateNetworks(); // generate networks for the current organisms
	bool UpdateGeneration(); // fitnesses should be set before calling this; returns true on success and false on failure

	int GetGenerationID() const; // for debugging
	int GetNumSpecies() const; // for debugging
	void PrintSpecieInfo() const; // for debugging

	int GetAddNodeNumber(std::pair<int, int> oldConnection, bool isRecurrent); // used by Genome::AddNodeMutation; shouldn't need to call this directly

private:
	class Organism {
	private:
		Genome genome;
	public:
		float fitness = -1; // gets set by test environment to a value >= 0
		Organism(const Genome& parent) : genome{ parent } {}
		Organism(std::ifstream& file);

		void Save(std::ofstream& file) const;

		const Genome& GetGenome() const { return genome; }

		bool operator<(const Organism& other) const {
			return fitness > other.fitness; // to sort by decreasing fitness
		}
	};

	struct Specie {
		std::vector<Organism> organisms;
		int specie_id = -1; // for debugging
		Specie() {}
		Specie(std::ifstream& file);

		void Save(std::ofstream& file) const;
	};

	NEAT(const NEAT&); // disable copy ctor (since there's no reason to copy, and we don't want to copy fitness_valid_ptr)
	std::shared_ptr<int> fitness_valid_ptr;

	bool WithinCompatibilityThresh(const Genome& g1, const Genome& g2) const;
	void AddGenome(std::vector<Specie>& newSpecies, const Genome& childGenome);

	int node_ctr = 0; // initialized in ctor
	std::map<std::pair<int, int>, int> forwardConnectNode; // map for getting node numbers when adding a new node
	std::map<std::pair<int, int>, int> recurrentConnectNode; // map for getting node numbers when adding a new node

	int species_ctr = -1;
	std::vector<Specie> species;

	int pop_size;
	float c1_c2;
	float c3;
	float compatibility_thresh;
	float top_p_cutoff;
	float add_node_mutation_prob;
	float add_edge_mutation_prob;
	float weight_mutation_prob;

	int generation_id = 0; // for debugging
};