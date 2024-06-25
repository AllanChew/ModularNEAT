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

#pragma comment(lib, "SDL2.lib")
#pragma comment(lib, "SDL2main.lib")
#pragma comment(lib, "SDL2_ttf.lib")

#include "SDL.h"
#include "SDL_ttf.h"
#include <iostream>
#include <string>
#include <sstream>
#include "./NEAT/Network.h"
#include "./NEAT/MathHelpers.h"
#include "XORTest.h"

static const int SCREEN_WIDTH = 1200;
static const int SCREEN_HEIGHT = 720;
static const int FONT_SIZE = 18; // 6
static const int NEURON_SIZE = 45; // 15
static const SDL_Color FONT_COLOR = { 0x00, 0x00, 0x00 };
static const int SCREEN_BORDER = 50;
static const float MOUSEWHEEL_SENSITIVITY = 0.01f;

static float CUR_MOUSEWHEEL_TOGGLE_VAL = 0.5; // gets clamped between 0 and 1

void DrawScaledLine(SDL_Renderer* gRenderer, int x1, int y1, int x2, int y2, float scale = 1) {
	SDL_RenderSetScale(gRenderer, scale, scale);
	SDL_RenderDrawLine(gRenderer, x1 / scale + 0.5f, y1 / scale + 0.5f, x2 / scale + 0.5f, y2 / scale + 0.5f);
	SDL_RenderSetScale(gRenderer, 1, 1);
}

void DrawScaledDot(SDL_Renderer* gRenderer, int x, int y, int width = 1, bool isOutline = false) {
	if (width < 2) {
		SDL_RenderDrawPoint(gRenderer, x, y);
		return;
	}

	int radius = width / 2;
	SDL_Rect fillRect = { x - radius, y - radius, 2 * radius + 1, 2 * radius + 1 };
	if (isOutline)
		SDL_RenderDrawRect(gRenderer, &fillRect);
	else
		SDL_RenderFillRect(gRenderer, &fillRect);
}

void DrawText(SDL_Renderer* gRenderer, TTF_Font* gFont, const char* text, SDL_Color color, int x_loc = 0, int y_loc = 0) {
	SDL_Surface* surface = TTF_RenderUTF8_Blended(gFont, text, color);
	if (surface == nullptr) {
		std::cerr << "Failed to render text surface: " << TTF_GetError() << std::endl;
		return;
	}

	SDL_Texture* mTexture = SDL_CreateTextureFromSurface(gRenderer, surface);
	if (mTexture == nullptr) {
		std::cerr << "Failed to create texture from rendered text: " << SDL_GetError() << std::endl;
		SDL_FreeSurface(surface);
		return;
	}

	SDL_Rect drawLoc = { x_loc - surface->w / 2, y_loc - surface->h / 2, surface->w, surface->h };
	SDL_FreeSurface(surface);
	SDL_RenderCopy(gRenderer, mTexture, NULL, &drawLoc);
	SDL_DestroyTexture(mTexture);
}

std::string int_to_string(int i) {
	std::ostringstream ss;
	ss << i;
	return ss.str();
}

void GetScreenPosn2(int layers, int layer_num, int layer_size, int layer_index, int& x_loc, int& y_loc) {
	y_loc = SCREEN_BORDER + (SCREEN_HEIGHT - SCREEN_BORDER * 2) * layer_index / layer_size;
	x_loc = SCREEN_BORDER + (SCREEN_WIDTH - SCREEN_BORDER * 2) * layer_num / layers;

	const int y_middle_offset = (SCREEN_HEIGHT - SCREEN_BORDER * 2) / (2 * layer_size);
	const int x_middle_offset = (SCREEN_WIDTH - SCREEN_BORDER * 2) / (2 * layers);

	y_loc += NEATMathHelpers::lerp(0, y_middle_offset * 2, 1 - CUR_MOUSEWHEEL_TOGGLE_VAL);
	x_loc += x_middle_offset;
}

void GetScreenPosn(const NetworkBaseVisual& n, const NeuronVisualInfo& v, int& x_loc, int& y_loc) {
	GetScreenPosn2(n.GetLayerSizes().size(), v.layer_num, v.is_output ? n.GetNumOutputNodes() : n.GetLayerSizes()[v.layer_num], v.layer_index, x_loc, y_loc);
}

void DrawNeuron(SDL_Renderer* gRenderer, TTF_Font* gFont, const NetworkBaseVisual& n, const NeuronVisualInfo& v) {
	int x_loc;
	int y_loc;
	GetScreenPosn(n, v, x_loc, y_loc);

	SDL_SetRenderDrawColor(gRenderer, 0xAF, 0xAF, 0xAF, 0xFF);
	DrawScaledDot(gRenderer, x_loc, y_loc, NEURON_SIZE);
	SDL_SetRenderDrawColor(gRenderer, 0x70, 0x70, 0x70, 0xFF);
	DrawScaledDot(gRenderer, x_loc, y_loc, NEURON_SIZE, true);

	std::string id_string = int_to_string(v.label);
	DrawText(gRenderer, gFont, id_string.c_str(), FONT_COLOR, x_loc, y_loc);
}

void DrawBezierCurve(SDL_Renderer* gRenderer, int x_start, int y_start, int x_end, int y_end, int x_start_vec, int y_start_vec, int x_end_vec, int y_end_vec, int points = 50) {
	int x_loc_prev = 0;
	int y_loc_prev = 0;
	for (int i = 0; i < points; ++i) {
		float alpha = float(i) / points;
		int x1 = NEATMathHelpers::lerp(x_start, x_start + x_start_vec, alpha);
		int y1 = NEATMathHelpers::lerp(y_start, y_start + y_start_vec, alpha);

		int x2 = NEATMathHelpers::lerp(x_end + x_end_vec, x_end, alpha);
		int y2 = NEATMathHelpers::lerp(y_end + y_end_vec, y_end, alpha);

		int x3 = NEATMathHelpers::lerp(x_start + x_start_vec, x_end + x_end_vec, alpha);
		int y3 = NEATMathHelpers::lerp(y_start + y_start_vec, y_end + y_end_vec, alpha);

		int x_loc = NEATMathHelpers::lerp(NEATMathHelpers::lerp(x1, x3, alpha), NEATMathHelpers::lerp(x3, x2, alpha), alpha);
		int y_loc = NEATMathHelpers::lerp(NEATMathHelpers::lerp(y1, y3, alpha), NEATMathHelpers::lerp(y3, y2, alpha), alpha);

		if (i != 0) SDL_RenderDrawLine(gRenderer, x_loc_prev, y_loc_prev, x_loc, y_loc);
		x_loc_prev = x_loc;
		y_loc_prev = y_loc;
	}

}

void DrawRecurrentEdge(SDL_Renderer* gRenderer, int x_loc1, int y_loc1, int x_loc2, int y_loc2) {
	SDL_SetRenderDrawColor(gRenderer, 0x9F, 0x9F, 0xFF, 0xFF);

	int y_multiplier = (y_loc2 < y_loc1) ? 1 : -1;

	if ((x_loc1 == x_loc2) && (y_loc1 == y_loc2)) // self connection
		DrawBezierCurve(gRenderer, x_loc1, y_loc1, x_loc2, y_loc2, 125, 90 * y_multiplier, -150, 50 * y_multiplier, 25);
	else
		DrawBezierCurve(gRenderer, x_loc1, y_loc1, x_loc2, y_loc2, 250, 180 * y_multiplier, -300, 100 * y_multiplier);
}

void DrawEdge(SDL_Renderer* gRenderer, TTF_Font* gFont, const NetworkBaseVisual& n, const NeuronVisualInfo& v1, const NeuronVisualInfo& v2, float weight) {
	int x_loc1;
	int y_loc1;
	int x_loc2;
	int y_loc2;
	GetScreenPosn(n, v1, x_loc1, y_loc1);
	GetScreenPosn(n, v2, x_loc2, y_loc2);

	if (v1.layer_num >= v2.layer_num) { // is recurrent
		DrawRecurrentEdge(gRenderer, x_loc1, y_loc1, x_loc2, y_loc2);
	}
	else {
		if (weight < 0) {
			SDL_SetRenderDrawColor(gRenderer, NEATMathHelpers::lerp(0x30, 0xFF, NEATMathHelpers::clamp(weight / -10)), 0x30, 0x30, 0xFF);
		}
		else {
			SDL_SetRenderDrawColor(gRenderer, 0x30, NEATMathHelpers::lerp(0x30, 0xFF, NEATMathHelpers::clamp(weight / 10)), 0x30, 0xFF);
		}
		DrawScaledLine(gRenderer, x_loc1, y_loc1, x_loc2, y_loc2, 1);
	}
}

void DrawNetwork(SDL_Renderer* gRenderer, TTF_Font* gFont, const NetworkBaseVisual& n) {
	// clear screen (white background)
	SDL_SetRenderDrawColor(gRenderer, 0xFF, 0xFF, 0xFF, 0xFF);
	SDL_RenderClear(gRenderer);

	// draw edges
	//std::vector<std::tuple<const NeuronVisualInfo*, const NeuronVisualInfo*, float>> edges = n.GetEdges();
	for (auto e : n.GetEdgesIterator()) {
		DrawEdge(gRenderer, gFont, n, *(std::get<0>(e)), *(std::get<1>(e)), std::get<2>(e));
	}

	// draw neurons
	for (auto& e : n.GetVisualInfo()) {
		DrawNeuron(gRenderer, gFont, n, e);
	}

	// update screen 
	SDL_RenderPresent(gRenderer);
}

int main(int argc, char* args[]) {
	// initialize SDL
	if (SDL_Init(SDL_INIT_VIDEO) < 0) {
		std::cerr << "Failed to initialize SDL: " << SDL_GetError() << std::endl;
		return 1;
	}

	if (TTF_Init() < 0) {
		std::cerr << "Failed to initialize SDL TTF: " << TTF_GetError() << std::endl;
		SDL_Quit();
		return 1;
	}

	// create window
	SDL_Window* gWindow = SDL_CreateWindow(
		"NEAT Visualizer",                 // window title
		SDL_WINDOWPOS_UNDEFINED,           // initial x position
		SDL_WINDOWPOS_UNDEFINED,           // initial y position
		SCREEN_WIDTH,                      // width, in pixels
		SCREEN_HEIGHT,                     // height, in pixels
		SDL_WINDOW_SHOWN
	);

	if (gWindow == nullptr) {
		std::cerr << "Failed to create window: " << SDL_GetError() << std::endl;
		SDL_Quit();
		TTF_Quit();
		return 1;
	}

	// create renderer for window
	SDL_Renderer* gRenderer = SDL_CreateRenderer(gWindow, -1, SDL_RENDERER_ACCELERATED);
	if (gRenderer == nullptr) {
		std::cerr << "Failed to create renderer: " << SDL_GetError() << std::endl;
		SDL_DestroyWindow(gWindow);
		SDL_Quit();
		TTF_Quit();
		return 1;
	}

	// create font for rendering text
	TTF_Font* gFont = TTF_OpenFont("font/arial.ttf", FONT_SIZE);
	if (gFont == nullptr) {
		std::cerr << "Failed to open font: " << TTF_GetError() << std::endl;
		SDL_DestroyRenderer(gRenderer);
		SDL_DestroyWindow(gWindow);
		SDL_Quit();
		TTF_Quit();
		return 1;
	}

	// finished setting up SDL, can now enter main loop

	XORTest xorTest;
	NetworkBaseVisual networkVisual;

	// to visualize a saved network when the program starts
	//networkVisual = NetworkBaseVisual("xor.dat");
	//DrawNetwork(gRenderer, gFont, networkVisual);

	bool bRunning = true;
	SDL_Event e;
	while (bRunning) {
		// handle events
		while (SDL_PollEvent(&e) != 0) {
			if (e.type == SDL_QUIT) bRunning = false;
			else if (e.type == SDL_MOUSEWHEEL) {
				CUR_MOUSEWHEEL_TOGGLE_VAL = NEATMathHelpers::clamp(CUR_MOUSEWHEEL_TOGGLE_VAL + e.wheel.y * MOUSEWHEEL_SENSITIVITY);
				DrawNetwork(gRenderer, gFont, networkVisual);
			}
			else if (e.type == SDL_KEYDOWN) {
				switch (e.key.keysym.sym) {
				case SDLK_RIGHT:
					networkVisual = xorTest.Tick();
					DrawNetwork(gRenderer, gFont, networkVisual);
					break;
				default:
					break;
				}
			}
		}
	}

	// cleanup SDL
	TTF_CloseFont(gFont);
	gFont = nullptr;
	SDL_DestroyRenderer(gRenderer);
	gRenderer = nullptr;
	SDL_DestroyWindow(gWindow);
	gWindow = nullptr;
	SDL_Quit();
	TTF_Quit();

	return 0;
}
