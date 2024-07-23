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
#include <unordered_set>
#include "Network.h"

class NEAT;

class Genome {
private:
	class Network : public NetworkBaseVisual {
	public:
		Network(int input_nodes, int output_nodes, const std::map<std::pair<int, int>, float>& forward_edges, const std::map<std::pair<int, int>, float>& recurrent_edges);

		void PrintForwardEdges() const; // for debugging

		bool FindNewPossibleConnection(int& in, int& out, bool& is_recurrent, int max_tries) const; // used by Genome::AddEdgeMutation

	protected:
		virtual void LoadImpl(std::ifstream& file) override;

	private:
		std::map<int, std::unordered_set<int>> adjacency_list; // used in FindNewPossibleConnection
		std::map<int, std::unordered_set<int>> adjacency_list_recurrent_rev; // used in FindNewPossibleConnection

		bool CheckRecurrent(int inputLabel, int outputLabel) const; // helper for FindNewPossibleConnection

		struct NeuronIdDepth {
			int id = 0;
			int depth = 0;
			NeuronIdDepth(int argID, int argDepth) : id{ argID }, depth{ argDepth } {}
			bool operator<(const NeuronIdDepth& other) const {
				if (depth == other.depth) return id < other.id; // same depth; so compare ID
				return depth < other.depth;
			}
		};
	};

public:
	Genome(int input_nodes, int output_nodes); // input nodes includes bias
	Genome(std::ifstream& file);
	void Save(std::ofstream& file) const;

	Network GenerateNetwork() const;

	bool AddNodeMutation(NEAT& n);
	bool AddEdgeMutation(const Network& network, float randomValStdDev, int max_tries = 3);
	void AddInputOutputEdge(float randomValStdDev);

	void Crossover(const Genome& parent1);

	void GetCompatibilityDistInfo(const Genome& genome, int& nonMatching_out, int& genomeSize_out, float& avgWeightDiff_out) const;

	void MutateWeights(float perturbStdDev, float randomValStdDev, float randomValProb);

private:
	int num_input_nodes;
	int num_output_nodes;
	std::map<std::pair<int, int>, float> forward_edges;
	std::map<std::pair<int, int>, float> recurrent_edges;

	std::map<std::pair<int, int>, float> disabled_forward_edges;
	std::map<std::pair<int, int>, float> disabled_recurrent_edges;

	bool IsOutputNode(int node_id) const;
};