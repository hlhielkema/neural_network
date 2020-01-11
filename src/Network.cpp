//  _    _ _      _ _          _    _ _      _ _                        
// | |  | (_)    | | |        | |  | (_)    | | |                       
// | |__| |_  ___| | | _____  | |__| |_  ___| | | _____ _ __ ___   __ _ 
// |  __  | |/ _ \ | |/ / _ \ |  __  | |/ _ \ | |/ / _ \ '_ ` _ \ / _` |
// | |  | | |  __/ |   <  __/ | |  | | |  __/ |   <  __/ | | | | | (_| |
// |_|  |_|_|\___|_|_|\_\___| |_|  |_|_|\___|_|_|\_\___|_| |_| |_|\__,_|
// _____________________________________________________________________
//
// Filename: Network.cpp
// Project: AI - Neural network
// Created: 26-06-2014 (DD-MM-YYYY)
// Changed: 27-06-2014 (DD-MM-YYYY)
//
// Author: Hielke Hielkema
// Contact: HielkeHielkema93@gmail.com
//
// (C) Hielke Hielkema - 2014

#include "stdafx.h"
#include "Network.h"

// Constructor
Network::Network(ActivationFunction * function, int inputsCount, int layerCount, int * neuronCounts)
{
	// Minimum of 1
	if (inputsCount < 1)
		inputsCount = 1;
	if (layerCount < 1)
		layerCount = 1;

	// Initialize the data
	this->m_inputCount = inputsCount;
	this->m_layerCount = layerCount;	
	this->m_output = NULL;
	this->m_layers = new Layer*[layerCount];
	for(int i = 0; i < layerCount; i++)
		this->m_layers[i] = new Layer(neuronCounts[i], 
								     (i == 0) ? inputsCount : neuronCounts[i - 1], 
									 function);
}

// Destructor
Network::~Network(void)
{
	for(int i = 0; i < this->m_layerCount; i++)
		delete this->m_layers[i];
	delete[] this->m_layers;
}

// Get the number of input values
int Network::getInputsCount()
{
	return this->m_inputCount;
}

// Get the number of layers
int Network::getLayersCount()
{
	return this->m_layerCount;
}

// Get the last output
double * Network::getOutput()
{
	return this->m_output;
}

// Get the layer with the given index
Layer * Network::getLayer(int index)
{
	return this->m_layers[index];
}

// Compute the output of the network
double * Network::compute(double * input)
{
	// Store the input in the output member
	this->m_output = input;

	// Compute each layer
	for(int i = 0; i < this->m_layerCount; i++)
		this->m_output = this->m_layers[i]->compute(this->m_output);

	// Return the output
	return this->m_output;
}

// Randomize the weights of all neurons in the network
void Network::randomize()
{
	for(int i = 0; i < this->m_layerCount; i++)
		this->m_layers[i]->randomize();
}

// Get the function of the first neuron of the first layer
ActivationFunction * Network::getFunction()
{
	return this->m_layers[0]->getNeuron(0)->getActivationFunction();
}