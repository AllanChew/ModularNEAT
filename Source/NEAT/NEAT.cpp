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

#include "NEAT.h"
#include "MathHelpers.h"
#include <algorithm>
#include <iostream>

NEAT::NEAT(int input_size, int output_size, int pop_size_in, float compatibility_thresh_in, float c1_c2_in, float c3_in, float top_p_cutoff_in, float add_node_mutation_prob_in, float add_edge_mutation_prob_in, float weight_mutation_prob_in)
	: fitness_valid_ptr{ std::make_shared<int>() },
	node_ctr{ input_size + output_size }, pop_size{ pop_size_in }, c1_c2{ c1_c2_in }, c3{ c3_in }, compatibility_thresh{ compatibility_thresh_in }, top_p_cutoff{ top_p_cutoff_in }, add_node_mutation_prob{ add_node_mutation_prob_in }, add_edge_mutation_prob{ add_edge_mutation_prob_in }, weight_mutation_prob{ weight_mutation_prob_in } {
	if (input_size <= 0) {
		std::cerr << "Failed to initialize NEAT due to invalid input size" << std::endl;
		return;
	}
	if (output_size <= 0) {
		std::cerr << "Failed to initialize NEAT due to invalid output size" << std::endl;
		return;
	}
	if (pop_size_in <= 0) {
		std::cerr << "Failed to initialize NEAT due to invalid population size" << std::endl;
		return;
	}

	const int input_nodes = input_size + 1; // + 1 for bias node
	const int output_nodes = output_size;

	const Genome emptyGenome = Genome(input_nodes, output_nodes);
	//auto emptyNetwork = emptyGenome.GenerateNetwork();

	// empty genome gets added as well since the initial add edge mutations might not cover all cases (e.g. it could all be the same edge)
	species.emplace_back();
	species.back().organisms.emplace_back(emptyGenome);
	species.back().specie_id = ++species_ctr;

	for (int organismsCreated = 1; organismsCreated < pop_size; ++organismsCreated) {
		Genome childGenome(input_nodes, output_nodes); // starts as empty genome
		childGenome.AddInputOutputEdge(2);
		//childGenome.AddEdgeMutation(emptyNetwork, 2); // initial add edge mutation

		bool foundSpecie = false;
		for (size_t j = 0; j < species.size(); ++j) {
			const Genome& representative = species[j].organisms[0].GetGenome();
			if (WithinCompatibilityThresh(representative, childGenome)) { // found which specie it belongs to
				species[j].organisms.emplace_back(childGenome);
				foundSpecie = true;
				break;
			}
		}
		if (!foundSpecie) { // new specie created
			species.emplace_back();
			species.back().organisms.emplace_back(childGenome);
			species.back().specie_id = ++species_ctr;
		}
	}
}

bool NEAT::WithinCompatibilityThresh(const Genome& g1, const Genome& g2) const {
	int nonMatching;
	int genomeSize;
	float avgWeightDiff;
	g1.GetCompatibilityDistInfo(g2, nonMatching, genomeSize, avgWeightDiff);

	const float compatibility_dist = (genomeSize <= 0) ? 0 : ((c1_c2 * nonMatching) / genomeSize + c3 * avgWeightDiff);
	return compatibility_dist < compatibility_thresh;
}

int NEAT::GetAddNodeNumber(std::pair<int, int> oldConnection, bool isRecurrent) {
	std::map<std::pair<int, int>, int>& connectNode = isRecurrent ? recurrentConnectNode : forwardConnectNode;

	auto it = connectNode.find(oldConnection);
	if (it != connectNode.end()) return it->second;

	connectNode[oldConnection] = ++node_ctr;
	return node_ctr;
}

void NEAT::AddGenome(std::vector<Specie>& newSpecies, const Genome& childGenome) {
	// check which specie child belongs to using compatibility threshold (may result in creation of new species)
	bool foundSpecie = false;
	for (size_t j = 0; j < newSpecies.size(); ++j) {
		const Genome& representative = (j < species.size()) ? species[j].organisms[0].GetGenome() : newSpecies[j].organisms[0].GetGenome(); // either most fit, or first of new specie
		if (WithinCompatibilityThresh(representative, childGenome)) { // found which specie it belongs to
			newSpecies[j].organisms.emplace_back(childGenome);
			foundSpecie = true;
			break;
		}
	}
	if (!foundSpecie) { // new specie created
		newSpecies.emplace_back();
		newSpecies.back().organisms.emplace_back(childGenome);
		newSpecies.back().specie_id = ++species_ctr;
	}
}

bool NEAT::UpdateGeneration() {
	// species for next generation
	// species from last generation are copied in, and new species get appended to the end
	// species from last generation that go extinct get removed later
	std::vector<Specie> newSpecies(species.size());
	for (size_t i = 0; i < species.size(); ++i) {
		newSpecies[i].specie_id = species[i].specie_id;
	}

	// specie fitnesses (fitness of organisms should've been set by testing environment)
	std::vector<float> specie_fitnesses(species.size(), 0);
	float specie_fitness_sum = 0;
	for (size_t i = 0; i < species.size(); ++i) {
		for (auto& organism : species[i].organisms) {
			if (organism.fitness < 0) { // should be set to a value >= 0
				std::cerr << "UpdateGeneration failed since not all fitnesses have been set yet" << std::endl;
				return false;
			}
			specie_fitnesses[i] += organism.fitness;
		}

		specie_fitnesses[i] /= species[i].organisms.size();
		specie_fitness_sum += specie_fitnesses[i];
	}

	if (specie_fitness_sum == 0) {
		std::cout << "UpdateGeneration Warning: specie_fitness_sum is equal to 0. Check fitness function." << std::endl;
	}

	for (auto& specie : species) {
		sort(specie.organisms.begin(), specie.organisms.end()); // sort by decreasing fitness
	}

	// create offspring (added into newSpecies)
	for (size_t i = 0; i < species.size(); ++i) {
		Specie& specie = species[i];
		int numOffspring = 0;
		if (specie_fitness_sum == 0) { // edge case where the fitness of every organism is 0 (this should be avoided in practice)
			numOffspring = pop_size / species.size();
		}
		else {
			numOffspring = pop_size * (specie_fitnesses[i] / specie_fitness_sum) + 0.5f;
		}

		if (numOffspring < 1) continue;

		// top organism (a.k.a. champion) of each specie is copied unchanged if numOffspring > 5
		if (numOffspring > 5) {
			AddGenome(newSpecies, specie.organisms[0].GetGenome());
			--numOffspring;
		}

		// top organisms used to create offspring (defaults to top 60%)
		int topOrganismsSize = specie.organisms.size() * top_p_cutoff + 0.5f;
		if (topOrganismsSize <= 0 || topOrganismsSize >= specie.organisms.size()) {
			topOrganismsSize = specie.organisms.size();
		}
		topOrganismsSize -= 1; // convert it into max index

		// create offspring
		for (int offspringCreated = 0; offspringCreated < numOffspring; ++offspringCreated) {
			int parent1_index = NEATMathHelpers::rand_int(topOrganismsSize);
			int parent2_index = NEATMathHelpers::rand_int(topOrganismsSize);

			// parent1 is the more fit parent (child inherits structure of more fit parent)
			//if (specie.organisms[parent2_index].fitness > specie.organisms[parent1_index].fitness) {
			if (parent1_index > parent2_index) { // since organisms have been sorted by decreasing fitness
				int temp_index = parent1_index;
				parent2_index = parent1_index;
				parent1_index = temp_index;
			}

			// cross-over parents to create child genome
			Genome childGenome = specie.organisms[parent1_index].GetGenome();
			if (parent1_index != parent2_index) childGenome.Crossover(specie.organisms[parent2_index].GetGenome()); // check index equality as an optimization

			// mutate child genome

			if (NEATMathHelpers::rand_norm() < add_node_mutation_prob) { // 3% chance by default
				childGenome.AddNodeMutation(*this); // add new node
			}
			else if (NEATMathHelpers::rand_norm() < add_edge_mutation_prob) { // 30% chance by default
				childGenome.AddEdgeMutation(childGenome.GenerateNetwork(), 2); // add new edge
			}
			else if (NEATMathHelpers::rand_norm() < weight_mutation_prob) { // 80% chance by default
				childGenome.MutateWeights(0.1f, 2.f, 0.1f); // mutate connection weights
			}

			AddGenome(newSpecies, childGenome); // save childGenome into newSpecies
		}

	}

	// update species
	species.clear();
	fitness_valid_ptr = std::make_shared<int>(); // make weak ptrs invalid
	for (int i = 0; i < newSpecies.size(); ++i) {
		const int numOrganisms = newSpecies[i].organisms.size();
		if (numOrganisms > 0) { // not empty (didn't go extinct)
			species.emplace_back(newSpecies[i]);
		}
	}

	return true;
}

std::vector<std::tuple<NetworkBaseVisual, FitnessInterface, int>> NEAT::GenerateNetworks() {
	std::vector<std::tuple<NetworkBaseVisual, FitnessInterface, int>> retVal;
	for (auto& specie : species) {
		for (auto& organism : specie.organisms) {
			retVal.emplace_back(organism.GetGenome().GenerateNetwork(), FitnessInterface(fitness_valid_ptr, organism.fitness), specie.specie_id);
		}
	}
	return retVal;
}

FitnessInterface::FitnessInterface(const std::shared_ptr<int>& fitness_valid_ptr_in, float& fitness_ref_in)
	: fitness_valid_ptr{ fitness_valid_ptr_in }, fitness_ref{ fitness_ref_in } {}

bool FitnessInterface::SetFitness(float f) {
	if (fitness_valid_ptr.expired()) { // if fitness_ref is no longer valid, it will print an error and do nothing
		std::cerr << "SetFitness failed since organism no longer exists. Make sure to call SetFitness before UpdateGeneration." << std::endl;
		return false;
	}
	if (f < 0) {
		std::cerr << "SetFitness failed since fitness value must be greater or equal to 0." << std::endl;
		return false;
	}
	fitness_ref = f;
	return true;
}

int NEAT::GetNumSpecies() const {
	return species.size();
}

void NEAT::PrintSpecieInfo() const {
	std::cout << "{SpecieID,SpecieSize}:";
	for (auto& e : species) {
		std::cout << " {" << e.specie_id << "," << e.organisms.size() << "}";
	}
	std::cout << std::endl;
}