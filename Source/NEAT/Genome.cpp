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

#include "Genome.h"
#include "MathHelpers.h"
#include "NEAT.h"

Genome::Genome(int input_nodes, int output_nodes) : num_input_nodes{ input_nodes }, num_output_nodes{ output_nodes } {}

void Genome::Crossover(const Genome& parent1) {
	for (auto& e : forward_edges) {
		auto edge_info = parent1.forward_edges.find(e.first);
		if (edge_info != parent1.forward_edges.end()) {
			if (NEATMathHelpers::rand_int(1) == 0) e.second = (edge_info->second);
		}
	}

	for (auto& e : recurrent_edges) {
		auto edge_info = parent1.recurrent_edges.find(e.first);
		if (edge_info != parent1.recurrent_edges.end()) {
			if (NEATMathHelpers::rand_int(1) == 0) e.second = (edge_info->second);
		}
	}
}

void Genome::GetCompatibilityDistInfo(const Genome& genome, int& nonMatching_out, int& genomeSize_out, float& avgWeightDiff_out) const {
	int matching = 0;
	float avgWeightDiff = 0;

	for (auto& e : forward_edges) {
		auto edge_info = genome.forward_edges.find(e.first);
		if (edge_info != genome.forward_edges.end()) {
			++matching;
			avgWeightDiff += abs(e.second - (edge_info->second));
		}
		else {
			edge_info = genome.disabled_forward_edges.find(e.first);
			if (edge_info != genome.disabled_forward_edges.end()) {
				++matching;
				avgWeightDiff += abs(e.second - (edge_info->second));
			}
		}
	}

	for (auto& e : disabled_forward_edges) {
		auto edge_info = genome.forward_edges.find(e.first);
		if (edge_info != genome.forward_edges.end()) {
			++matching;
			avgWeightDiff += abs(e.second - (edge_info->second));
		}
		else {
			edge_info = genome.disabled_forward_edges.find(e.first);
			if (edge_info != genome.disabled_forward_edges.end()) {
				++matching;
				avgWeightDiff += abs(e.second - (edge_info->second));
			}
		}
	}

	for (auto& e : recurrent_edges) {
		auto edge_info = genome.recurrent_edges.find(e.first);
		if (edge_info != genome.recurrent_edges.end()) {
			++matching;
			avgWeightDiff += abs(e.second - (edge_info->second));
		}
		else {
			edge_info = genome.disabled_recurrent_edges.find(e.first);
			if (edge_info != genome.disabled_recurrent_edges.end()) {
				++matching;
				avgWeightDiff += abs(e.second - (edge_info->second));
			}
		}
	}

	for (auto& e : disabled_recurrent_edges) {
		auto edge_info = genome.recurrent_edges.find(e.first);
		if (edge_info != genome.recurrent_edges.end()) {
			++matching;
			avgWeightDiff += abs(e.second - (edge_info->second));
		}
		else {
			edge_info = genome.disabled_recurrent_edges.find(e.first);
			if (edge_info != genome.disabled_recurrent_edges.end()) {
				++matching;
				avgWeightDiff += abs(e.second - (edge_info->second));
			}
		}
	}

	const int nonMatching = forward_edges.size() + disabled_forward_edges.size() + genome.forward_edges.size() + genome.disabled_forward_edges.size() +
		recurrent_edges.size() + disabled_recurrent_edges.size() + genome.recurrent_edges.size() + genome.disabled_recurrent_edges.size() - 2 * matching;
	nonMatching_out = nonMatching;
	genomeSize_out = nonMatching + matching;
	avgWeightDiff_out = (matching == 0) ? 0 : (avgWeightDiff / matching);
}

void Genome::MutateWeights(float perturbStdDev, float randomValStdDev, float randomValProb) {
	for (auto& e : forward_edges) {
		if (NEATMathHelpers::rand_norm() < randomValProb) e.second = NEATMathHelpers::randomGaussian(randomValStdDev);
		else e.second += NEATMathHelpers::randomGaussian(perturbStdDev);
	}

	for (auto& e : recurrent_edges) {
		if (NEATMathHelpers::rand_norm() < randomValProb) e.second = NEATMathHelpers::randomGaussian(randomValStdDev);
		else e.second += NEATMathHelpers::randomGaussian(perturbStdDev);
	}
}

bool Genome::IsOutputNode(int node_id) const {
	return !((node_id < num_input_nodes) || (node_id >= (num_input_nodes + num_output_nodes)));
}

Genome::Network Genome::GenerateNetwork() const {
	return Network(num_input_nodes, num_output_nodes, forward_edges, recurrent_edges);
}

bool Genome::AddNodeMutation(NEAT& n) {
	std::vector<std::pair<int, int>> possibleEdges;

	for (auto& e : forward_edges) {
		auto fromNode = std::get<0>(e.first);
		if (IsOutputNode(fromNode)) continue;

		possibleEdges.emplace_back(e.first);
	}

	const int numNormalEdges = possibleEdges.size();

	for (auto& e : recurrent_edges) {
		auto fromNode = std::get<0>(e.first);
		if (IsOutputNode(fromNode)) continue;

		possibleEdges.emplace_back(e.first);
	}

	if (possibleEdges.size() < 1) return false; // no possible edges to split

	const int randIndex = NEATMathHelpers::rand_int(possibleEdges.size() - 1);
	const bool isRecurrent = randIndex >= numNormalEdges;
	const int oldFromNode = std::get<0>(possibleEdges[randIndex]);
	const int oldToNode = std::get<1>(possibleEdges[randIndex]);
	const int newNode = n.GetAddNodeNumber(possibleEdges[randIndex], isRecurrent);

	float oldWeight;
	if (isRecurrent) {
		oldWeight = recurrent_edges[possibleEdges[randIndex]];
		recurrent_edges.erase(possibleEdges[randIndex]);
		disabled_recurrent_edges[possibleEdges[randIndex]] = oldWeight;
	}
	else {
		oldWeight = forward_edges[possibleEdges[randIndex]];
		forward_edges.erase(possibleEdges[randIndex]);
		disabled_forward_edges[possibleEdges[randIndex]] = oldWeight;
	}

	forward_edges[{oldFromNode, newNode}] = 1;
	if (isRecurrent) {
		recurrent_edges[{newNode, oldToNode}] = oldWeight;
	}
	else {
		forward_edges[{newNode, oldToNode}] = oldWeight;
	}

	return true;
}

// should be used on empty genome
void Genome::AddInputOutputEdge(float randomValStdDev) {
	int in = NEATMathHelpers::rand_int(num_input_nodes - 1); // any input node (or bias)
	int out = NEATMathHelpers::rand_int(num_input_nodes, num_input_nodes + num_output_nodes - 1); // any output node

	forward_edges[{in, out}] = NEATMathHelpers::randomGaussian(randomValStdDev);
}

// this could enable a disabled connection
bool Genome::AddEdgeMutation(const Network& network, float randomValStdDev, int max_tries) {
	int in;
	int out;
	bool is_recurrent;

	if (!network.FindNewPossibleConnection(in, out, is_recurrent, max_tries)) return false;

	if (is_recurrent) {
		recurrent_edges[{in, out}] = NEATMathHelpers::randomGaussian(randomValStdDev);
		disabled_recurrent_edges.erase({ in,out });
	}
	else {
		forward_edges[{in, out}] = NEATMathHelpers::randomGaussian(randomValStdDev);
		disabled_forward_edges.erase({ in,out });
	}

	return true;
}