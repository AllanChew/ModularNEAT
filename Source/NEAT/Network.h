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

#include <vector>
#include <memory>
#include <iostream>

// struct for holding visualization information of a neuron
struct NeuronVisualInfo {
	int label = 0;
	int layer_num = 0; // the layer that the neuron belongs to (0 being input layer)
	int layer_index = 0; // the index of the neuron within the layer or the output index if is_output == true
	bool is_output = false;
	NeuronVisualInfo(int argLabel, int argLayerNum, int argLayerIndex, bool argIsOutput)
		: label{ argLabel }, layer_num{ argLayerNum }, layer_index{ argLayerIndex }, is_output{ argIsOutput } {}
	NeuronVisualInfo() {}
};

// lightweight network for processing information
class NetworkBase {
public:
	NetworkBase();
	NetworkBase(const char* fname);
	virtual ~NetworkBase() {}

	void ResetRecurrentConnections();

	bool IsInvalid() const; // can be used to check if ctor successfully loaded from file

	int GetNumNodes() const; // for debugging
	int GetNumEdges() const; // for debugging
	int GetNumOutputNodes() const; // for debugging and also used for visualization

protected:
	struct NeuronInputInfo {
		int input_index = 0;
		float weight = 0;
		NeuronInputInfo(int argInputIndex, float argWeight) : input_index{ argInputIndex }, weight{ argWeight } {}
		NeuronInputInfo() {}
	};

	struct NeuronRunInfo {
		float output_val = 0;
		int input_info_block_size = 0;
		NeuronRunInfo(float argOutputVal, int argInputSize)
			: output_val{ argOutputVal }, input_info_block_size{ argInputSize } {}
		NeuronRunInfo() {}
	};

	int num_input_nodes = 0;
	int num_output_nodes = 0;

	bool IsOutputNode(int node_id) const;
	bool IsInputNode(int node_id) const;

	void Load(const char* fname);
	virtual void LoadImpl(std::ifstream& file);

	std::shared_ptr<std::vector<NeuronInputInfo>> input_info; // shared_ptr so that copied networks point to the same weights vector
	std::shared_ptr<std::vector<int>> output_indices;
	std::vector<NeuronRunInfo> run_info;

public:
	template<typename T, typename U>
	bool Run(const std::vector<T>& in, std::vector<U>& out) {
		if (IsInvalid()) {
			std::cerr << "Run failed since NetworkBase is corrupted or hasn't been initialized" << std::endl;
			return false;
		}

		if (in.size() != (num_input_nodes - 1)) {
			std::cerr << "NetworkBase::Run received input vector with incorrect size" << std::endl;
			return false;
		}

		if (out.size() != num_output_nodes) {
			std::cerr << "NetworkBase::Run received output vector with incorrect size" << std::endl;
			return false;
		}

		for (int i = 0; i < (num_input_nodes - 1); ++i) {
			run_info[i].output_val = in[i];
		}

		run_info[num_input_nodes - 1].output_val = 1; // bias always set to 1

		int input_info_start_index = 0;
		for (int i = num_input_nodes; i < run_info.size(); ++i) {
			const int numPrevNodes = run_info[i].input_info_block_size;
			float sum = 0;
			if (numPrevNodes > 0) {
				const NeuronInputInfo* prevInfo = &((*input_info)[input_info_start_index]);

				for (int j = 0; j < numPrevNodes; ++j) {
					sum += run_info[prevInfo[j].input_index].output_val * prevInfo[j].weight;
				}
				input_info_start_index += numPrevNodes;
			}
			run_info[i].output_val = tanh(sum); // currently hardcoding tanh as activation for all neurons
		}

		for (int i = 0; i < num_output_nodes; ++i) {
			out[i] = run_info[(*output_indices)[i]].output_val;
		}

		return true;
	}
};

// NetworkBase extended with visualization information
class NetworkBaseVisual : public NetworkBase {
public:
	class iterator {
	public:
		iterator(const NetworkBaseVisual* ptr, bool isEnd);
		iterator begin() const;
		iterator end() const;
		iterator operator++();
		bool operator!=(const iterator& other) const;
		std::tuple<const NeuronVisualInfo*, const NeuronVisualInfo*, float> operator*() const;
	private:
		void validate_node();
		int node_index = 0;
		int prev_index = 0;
		int input_info_start_index = 0;
		bool isEnd = false;
		const NetworkBaseVisual* ptr = nullptr;
	};

	NetworkBaseVisual();
	NetworkBaseVisual(const char* fname);

	bool Save(const char* fname) const; // returns true on success and false on failure

	const std::vector<NeuronVisualInfo>& GetVisualInfo() const;
	const std::vector<int>& GetLayerSizes() const;

	iterator GetEdgesIterator() const;
	//std::vector<std::tuple<const NeuronVisualInfo*, const NeuronVisualInfo*, float>> GetEdges() const;

protected:
	virtual void LoadImpl(std::ifstream& file) override;

	std::vector<NeuronVisualInfo> visual_info; // contains labels which is used for visualization as well as finding possible connections for NEAT
	std::vector<int> layer_sizes;
};
