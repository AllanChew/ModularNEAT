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
#include <fstream>

namespace NEATSerializeMap {
	template<typename T, typename U>
	void SaveMap(const std::map<std::pair<T, T>, U>& inMap, std::ofstream& file) {
		int mapSize = inMap.size();
		file.write((const char*)(&mapSize), sizeof(int));

		for (auto& e : inMap) {
			file.write((const char*)(&std::get<0>(e.first)), sizeof(T));
			file.write((const char*)(&std::get<1>(e.first)), sizeof(T));
			file.write((const char*)(&e.second), sizeof(U));
		}
	}

	template<typename T, typename U>
	void LoadMap(std::map<std::pair<T, T>, U>& inMap, std::ifstream& file, bool resetBeforeLoad = true) {
		if (resetBeforeLoad) inMap.clear();

		int mapSize;
		file.read((char*)(&mapSize), sizeof(int));

		T fromNode;
		T toNode;
		U weight;
		for (int i = 0; i < mapSize; ++i) {
			file.read((char*)(&fromNode), sizeof(T));
			file.read((char*)(&toNode), sizeof(T));
			file.read((char*)(&weight), sizeof(U));
			inMap[{fromNode, toNode}] = weight;
		}
	}
}