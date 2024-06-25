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

#include "Network.h"
#include "Genome.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include "MathHelpers.h"
#include <deque>

bool NetworkBase::IsInputNode(int node_id) const {
	return !((node_id < 0) || (node_id >= num_input_nodes));
}

bool NetworkBase::IsOutputNode(int node_id) const {
	return !((node_id < num_input_nodes) || (node_id >= (num_input_nodes + num_output_nodes)));
}

bool NetworkBase::IsInvalid() const {
	return num_input_nodes < 2 || num_output_nodes < 1;
}

NetworkBase::NetworkBase()
	: num_input_nodes{ 0 }, num_output_nodes{ 0 },
	input_info{ std::make_shared<std::vector<NeuronInputInfo>>() },
	output_indices{ std::make_shared<std::vector<int>>() } {}

NetworkBaseVisual::NetworkBaseVisual() {}

NetworkBase::NetworkBase(const char* fname) { Load(fname); }

NetworkBaseVisual::NetworkBaseVisual(const char* fname) { Load(fname); }

int NetworkBase::GetNumNodes() const {
	return run_info.size();
}

int NetworkBase::GetNumEdges() const {
	return input_info->size();
}

bool NetworkBaseVisual::Save(const char* fname) const {
	if (IsInvalid()) {
		std::cerr << "Save failed since NetworkBaseVisual is corrupted or hasn't been initialized" << std::endl;
		return false;
	}

	std::ofstream file{ fname, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc };

	// save network base info
	const int input_info_size = (*input_info).size();
	const int output_indices_size = (*output_indices).size();
	const int run_info_size = run_info.size();

	file.write((const char*)(&num_input_nodes), sizeof(int));
	file.write((const char*)(&num_output_nodes), sizeof(int));
	file.write((const char*)(&input_info_size), sizeof(int));
	file.write((const char*)(&output_indices_size), sizeof(int));
	file.write((const char*)(&run_info_size), sizeof(int));

	file.write((const char*)(&(*input_info)[0]), sizeof(NeuronInputInfo) * input_info_size);
	file.write((const char*)(&(*output_indices)[0]), sizeof(int) * output_indices_size);
	file.write((const char*)(&run_info[0]), sizeof(NeuronRunInfo) * run_info_size);

	// save visualization info
	const int visual_info_size = visual_info.size();
	const int layers = layer_sizes.size();

	file.write((const char*)(&visual_info_size), sizeof(int));
	file.write((const char*)(&layers), sizeof(int));

	file.write((const char*)(&visual_info[0]), sizeof(NeuronVisualInfo) * visual_info_size);
	file.write((const char*)(&layer_sizes[0]), sizeof(int) * layers);

	file.close();

	return true;
}

void NetworkBase::Load(const char* fname) {
	std::ifstream file{ fname, std::ios::binary };
	if (!file.is_open()) {
		std::cerr << "Failed to open " << fname << std::endl;
		return;
	}

	LoadImpl(file);

	file.close();
}

void NetworkBase::LoadImpl(std::ifstream& file) {
	// create new shared ptrs so that other networks don't get corrupted
	input_info = std::make_shared<std::vector<NeuronInputInfo>>();
	output_indices = std::make_shared<std::vector<int>>();

	int input_info_size;
	int output_indices_size;
	int run_info_size;

	file.read((char*)(&num_input_nodes), sizeof(int));
	file.read((char*)(&num_output_nodes), sizeof(int));
	file.read((char*)(&input_info_size), sizeof(int));
	file.read((char*)(&output_indices_size), sizeof(int));
	file.read((char*)(&run_info_size), sizeof(int));

	(*input_info).resize(input_info_size);
	(*output_indices).resize(output_indices_size);
	run_info.resize(run_info_size);

	file.read((char*)(&(*input_info)[0]), sizeof(NeuronInputInfo) * input_info_size);
	file.read((char*)(&(*output_indices)[0]), sizeof(int) * output_indices_size);
	file.read((char*)(&run_info[0]), sizeof(NeuronRunInfo) * run_info_size);

	ResetRecurrentConnections();
}

void NetworkBaseVisual::LoadImpl(std::ifstream& file) {
	NetworkBase::LoadImpl(file);

	int visual_info_size;
	int layers;

	file.read((char*)(&visual_info_size), sizeof(int));
	file.read((char*)(&layers), sizeof(int));

	visual_info.resize(visual_info_size);
	layer_sizes.resize(layers);

	file.read((char*)(&visual_info[0]), sizeof(NeuronVisualInfo) * visual_info_size);
	file.read((char*)(&layer_sizes[0]), sizeof(int) * layers);
}

void Genome::Network::LoadImpl(std::ifstream& file) {
	std::cerr << "Load failed. Do not use the Network class; use NetworkBaseVisual or NetworkBase instead." << std::endl;
}

// input_nodes must be >= 2 (we need at least one input to be useful, and an extra is used as a bias)
// output_nodes must be >= 1
Genome::Network::Network(int input_nodes, int output_nodes, const std::map<std::pair<int, int>, float>& forward_edges, const std::map<std::pair<int, int>, float>& recurrent_edges) {
	num_input_nodes = input_nodes;
	num_output_nodes = output_nodes;

	// read into adjacency lists
	//std::map<int,std::unordered_set<int>> adjacency_list; // now a member variable
	std::map<int, std::unordered_set<int>> adjacency_list_rev; // gets mutated below during topological sort
	for (int i = 0; i < (input_nodes + output_nodes); ++i) {
		adjacency_list[i];
		adjacency_list_rev[i];
	}
	for (auto& e : forward_edges) {
		auto fromNode = std::get<0>(e.first);
		auto toNode = std::get<1>(e.first);

		if (IsOutputNode(fromNode) && !IsOutputNode(toNode)) {
			std::cerr << "Found output to non-output edge that isn't labelled as recurrent!" << std::endl;
			return;
		}

		adjacency_list[fromNode].insert(toNode);
		adjacency_list[toNode];
		adjacency_list_rev[toNode].insert(fromNode);
		adjacency_list_rev[fromNode];
	}

	std::map<int, std::unordered_set<int>> adjacency_list_rev_copy = adjacency_list_rev; // make a copy for after topological sort

	// topological sort
	std::vector<int> sortedNodes;
	for (int i = 0; i < input_nodes; ++i) { // add input nodes first (always a source)
		sortedNodes.emplace_back(i);
	}
	for (auto& e : adjacency_list_rev) {
		if (IsInputNode(e.first)) continue; // input nodes already added (always added first)
		if (IsOutputNode(e.first)) continue; // output nodes handled later
		if (e.second.size() == 0) sortedNodes.emplace_back(e.first); // found a source node
	}
	int sortedNodesIndex = 0;
	for (; sortedNodesIndex < sortedNodes.size(); ++sortedNodesIndex) {
		int curNode = sortedNodes[sortedNodesIndex];
		for (auto& e : adjacency_list[curNode]) {
			adjacency_list_rev[e].erase(curNode);
			if (adjacency_list_rev[e].size() <= 0 && !IsOutputNode(e))
				sortedNodes.emplace_back(e);
		}
	}

	// now topological sort the outputs
	for (int i = 0; i < output_nodes; ++i) { // add output source nodes
		int curNode = input_nodes + i;
		if (adjacency_list_rev[curNode].size() <= 0) sortedNodes.emplace_back(curNode); // found output source node
	}
	for (; sortedNodesIndex < sortedNodes.size(); ++sortedNodesIndex) {
		int curNode = sortedNodes[sortedNodesIndex];
		if (!IsOutputNode(curNode)) { // sanity check
			std::cerr << "Expected an output node!" << std::endl; // failed to label output to non-output node as recurrent
			return;
		}
		for (auto& e : adjacency_list[curNode]) {
			adjacency_list_rev[e].erase(curNode);
			if (adjacency_list_rev[e].size() <= 0) {
				sortedNodes.emplace_back(e);
			}
		}
	}

	// calculate max depth
	std::map<int, int> maxDepth;
	int outputDepth = 0;
	for (int i = 0; i < sortedNodes.size(); ++i) {
		int curNode = sortedNodes[i];
		if (IsInputNode(curNode)) {
			maxDepth[curNode] = 0; // input is given maxDepth of 0
			continue;
		}

		int curMaxDepth = 1;
		if (IsOutputNode(curNode)) curMaxDepth = outputDepth + 1;
		for (auto& e : adjacency_list_rev_copy[curNode]) {
			const int newDepth = maxDepth[e] + 1;
			if (newDepth > curMaxDepth) curMaxDepth = newDepth;
		}
		maxDepth[curNode] = curMaxDepth;

		if (!IsOutputNode(curNode) && (curMaxDepth > outputDepth)) outputDepth = curMaxDepth;
	}

	std::vector<NeuronIdDepth> id_depth_pair;
	id_depth_pair.reserve(maxDepth.size());
	for (auto& e : maxDepth) {
		id_depth_pair.emplace_back(e.first, e.second);
		//std::cout << e.first << ": " << e.second << std::endl;
	}
	sort(id_depth_pair.begin(), id_depth_pair.end());
	/*for (auto& e : id_depth_pair) {
		std::cout << e.first << ": " << e.second << std::endl;
	}*/

	// can now start flattening into internal indices

	visual_info.reserve(id_depth_pair.size());

	int last_depth = 0;
	int curIndex = 0;
	for (auto& e : id_depth_pair) {
		if (e.depth != last_depth) {
			layer_sizes.emplace_back(curIndex);
			last_depth = e.depth;
			curIndex = 0;
		}
		const bool bIsOutputNode = IsOutputNode(e.id);
		visual_info.emplace_back(e.id, e.depth, bIsOutputNode ? (e.id - input_nodes) : curIndex, bIsOutputNode);
		++curIndex;
	}
	layer_sizes.emplace_back(curIndex);

	// mapping from label/id to internal index
	std::map<int, int> internal_index_map;
	for (int i = 0; i < id_depth_pair.size(); ++i) {
		internal_index_map[id_depth_pair[i].id] = i;
	}

	// std::map<int,std::unordered_set<int>> adjacency_list_recurrent_rev; // now a member variable
	for (auto& e : recurrent_edges) {
		auto fromNode = std::get<0>(e.first);
		auto toNode = std::get<1>(e.first);
		adjacency_list_recurrent_rev[toNode].insert(fromNode);
		adjacency_list_recurrent_rev[fromNode];
	}

	output_indices->reserve(num_output_nodes);
	for (int i = 0; i < num_output_nodes; ++i) {
		output_indices->emplace_back(0);
	}

	run_info.reserve(id_depth_pair.size());
	for (int i = 0; i < id_depth_pair.size(); ++i) {
		int curNode = id_depth_pair[i].id;

		if (IsOutputNode(curNode)) (*output_indices)[curNode - num_input_nodes] = i;

		for (auto& f : adjacency_list_rev_copy[curNode]) {
			auto edge_info = forward_edges.find({ f, curNode });
			float weight = 0;
			if (edge_info != forward_edges.end()) weight = (edge_info->second);
			input_info->emplace_back(internal_index_map[f], weight);
		}
		for (auto& f : adjacency_list_recurrent_rev[curNode]) {
			auto edge_info = recurrent_edges.find({ f, curNode });
			float weight = 0;
			if (edge_info != recurrent_edges.end()) weight = (edge_info->second);
			input_info->emplace_back(internal_index_map[f], weight);
		}

		int input_info_block_size = adjacency_list_rev_copy[curNode].size() + adjacency_list_recurrent_rev[curNode].size();
		run_info.emplace_back((float)(0), input_info_block_size);
	}

	// set output of bias to 1
	run_info[input_nodes - 1].output_val = 1;

	/*
	for (auto& e : adjacency_list) {
		std::cout << e.first << std::endl;
		std::cout << " IsOutput: " << IsOutputNode(e.first)
			<< ", IsInput: " << IsInputNode(e.first) << std::endl;
	}

	std::cout << "forward edges" << std::endl;
	for (auto &e : adjacency_list) {
		std::cout << e.first << std::endl;
		for (auto &f: e.second) {
			std::cout << " " << f << std::endl;
		}
	}
	std::cout << "reverse edges" << std::endl;
	for (auto &e : adjacency_list_rev) {
		std::cout << e.first << std::endl;
		for (auto &f: e.second) {
			std::cout << " " << f << std::endl;
		}
	}
	*/
}

/*
std::vector<std::tuple<const NeuronVisualInfo *, const NeuronVisualInfo *, float>> NetworkBaseVisual::GetEdges() const {
	std::vector<std::tuple<const NeuronVisualInfo *, const NeuronVisualInfo *, float>> ret_val;

	int input_info_start_index = 0;
	for (int i = 0; i < run_info.size(); ++i) {
		const int numPrevNodes = run_info[i].input_info_block_size;
		if (numPrevNodes < 1) continue;

		const NeuronInputInfo *prevInfo = &((*input_info)[input_info_start_index]);
		for (int j = 0; j < numPrevNodes; ++j) {
			ret_val.emplace_back(&visual_info[prevInfo[j].input_index], &visual_info[i], prevInfo[j].weight);
		}
		input_info_start_index += numPrevNodes;
	}
	return ret_val;
}
*/

const std::vector<NeuronVisualInfo>& NetworkBaseVisual::GetVisualInfo() const {
	return visual_info;
}

const std::vector<int>& NetworkBaseVisual::GetLayerSizes() const {
	return layer_sizes;
}

int NetworkBase::GetNumOutputNodes() const {
	return num_output_nodes;
}

void Genome::Network::PrintForwardEdges() const {
	for (auto& e : adjacency_list) {
		std::cout << e.first << std::endl;
		for (auto& f : e.second) {
			std::cout << " " << f << std::endl;
		}
	}
}

bool Genome::Network::CheckRecurrent(int inputLabel, int outputLabel) const {
	if (inputLabel == outputLabel) return true;
	if (IsOutputNode(inputLabel) && !IsOutputNode(outputLabel)) return true;

	// start at outputLabel and see if we can find inputLabel using BFS
	std::unordered_set<int> discovered;
	std::deque<int> frontier;
	discovered.insert(outputLabel);
	frontier.push_back(outputLabel);
	while (frontier.size() > 0) {
		int curNode = frontier.front();
		frontier.pop_front();

		if (curNode == inputLabel) return true; // found input label

		auto forwardEdgeLookup = adjacency_list.find(curNode);
		if (forwardEdgeLookup == adjacency_list.end()) {
			std::cerr << "Could not find node " << curNode << " in network" << std::endl;
			return true;
		}

		for (auto& e : forwardEdgeLookup->second) {
			if (discovered.count(e) < 1) {
				discovered.insert(e);
				frontier.push_back(e);
			}
		}
	}

	return false;
}

bool Genome::Network::FindNewPossibleConnection(int& in, int& out, bool& is_recurrent, int max_tries) const {
	for (int try_num = 0; try_num < max_tries; ++try_num) {
		int randInput = NEATMathHelpers::rand_int(run_info.size() - 1); // can be any node
		int randInputLabel = visual_info[randInput].label;
		int randOutput = NEATMathHelpers::rand_int(num_input_nodes, run_info.size() - 1); // any node that isn't an input (or bias)
		int randOutputLabel = visual_info[randOutput].label;

		// check if connection already exists (check normal and recurrent connections)
		auto forwardEdgeLookup = adjacency_list.find(randInputLabel);
		if (forwardEdgeLookup == adjacency_list.end()) {
			std::cerr << "Could not find node " << randInputLabel << std::endl;
			continue;
		}
		if (forwardEdgeLookup->second.count(randOutputLabel) > 0) continue; // edge exists

		auto recurrentEdgeLookup = adjacency_list_recurrent_rev.find(randOutputLabel);
		if (recurrentEdgeLookup != adjacency_list_recurrent_rev.end()) {
			if (recurrentEdgeLookup->second.count(randInputLabel) > 0) continue; // edge exists
		}

		in = randInputLabel;
		out = randOutputLabel;
		is_recurrent = CheckRecurrent(randInputLabel, randOutputLabel);
		return true;
	}

	return false;
}

void NetworkBase::ResetRecurrentConnections() {
	for (auto& e : run_info) {
		e.output_val = 0; // resets all neuron outputs to 0
	}
}

NetworkBaseVisual::iterator NetworkBaseVisual::GetEdgesIterator() const {
	return NetworkBaseVisual::iterator(this, false); // is at beginning
}

NetworkBaseVisual::iterator::iterator(const NetworkBaseVisual* ptr, bool isEnd) : isEnd{ isEnd }, ptr{ ptr } {
	if (!isEnd) validate_node();
}

NetworkBaseVisual::iterator NetworkBaseVisual::iterator::begin() const {
	return NetworkBaseVisual::iterator(ptr, false);
}

NetworkBaseVisual::iterator NetworkBaseVisual::iterator::end() const {
	return NetworkBaseVisual::iterator(ptr, true);
}

void NetworkBaseVisual::iterator::validate_node() {
	// find next node with an edge
	for (; node_index < ptr->run_info.size(); ++node_index) {
		const int numPrevNodes = ptr->run_info[node_index].input_info_block_size;
		if (numPrevNodes >= 1) return; // found node with an edge
	}
	isEnd = true; // no edges
}

NetworkBaseVisual::iterator NetworkBaseVisual::iterator::operator++() {
	if (isEnd) return *this;

	const int numPrevNodes = ptr->run_info[node_index].input_info_block_size;
	++prev_index;
	if (prev_index < numPrevNodes) return *this; // we're done

	input_info_start_index += numPrevNodes;
	++node_index;
	prev_index = 0;
	validate_node();

	return *this;
}

bool NetworkBaseVisual::iterator::operator!=(const iterator& other) const {
	if (ptr != other.ptr) return true;

	if (isEnd && other.isEnd) return false;

	if (isEnd != other.isEnd) return true;

	// else both haven't ended

	return (node_index != other.node_index) || (prev_index != other.prev_index);
}

std::tuple<const NeuronVisualInfo*, const NeuronVisualInfo*, float> NetworkBaseVisual::iterator::operator*() const {
	const NeuronInputInfo* prevInfo = &((*(ptr->input_info))[input_info_start_index]);
	return { &(ptr->visual_info[prevInfo[prev_index].input_index]), &(ptr->visual_info[node_index]), prevInfo[prev_index].weight };
}
