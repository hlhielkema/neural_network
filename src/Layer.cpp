//  _    _ _      _ _          _    _ _      _ _                        
// | |  | (_)    | | |        | |  | (_)    | | |                       
// | |__| |_  ___| | | _____  | |__| |_  ___| | | _____ _ __ ___   __ _ 
// |  __  | |/ _ \ | |/ / _ \ |  __  | |/ _ \ | |/ / _ \ '_ ` _ \ / _` |
// | |  | | |  __/ |   <  __/ | |  | | |  __/ |   <  __/ | | | | | (_| |
// |_|  |_|_|\___|_|_|\_\___| |_|  |_|_|\___|_|_|\_\___|_| |_| |_|\__,_|
// _____________________________________________________________________
//
// Filename: Layer.cpp
// Project: AI - Neural network
// Created: 26-06-2014 (DD-MM-YYYY)
// Changed: 27-06-2014 (DD-MM-YYYY)
//
// Author: Hielke Hielkema
// Contact: HielkeHielkema93@gmail.com
//
// (C) Hielke Hielkema - 2014

#include "Layer.h"

// Constructor
Layer::Layer(int neurons, int inputs, ActivationFunction * function)
{
	// Minimum of 1
	if (neurons < 1)
		neurons = 1;
	if (inputs < 1)
		inputs = 1;

	// Initalize the data
	this->m_inputCount = inputs;
	this->m_neuronCount = neurons;

	this->m_neurons = new Neuron*[neurons];
	this->m_output = new double[neurons];

	// Create each neuron
	for(int i = 0; i < neurons; i++)
		this->m_neurons[i] = new Neuron(inputs, function);
}

// Destructor
Layer::~Layer(void)
{
	for(int i = 0; i < this->m_neuronCount; i++)
		delete this->m_neurons[i];
	delete[] this->m_neurons;
	delete[] this->m_output;
}

// Get the number of input values
int Layer::getInputsCount()
{
	return this->m_inputCount;
}

// Get the number of neurons
int Layer::getNeuronsCount()
{
	return this->m_neuronCount;
}

// Get the layer output
double * Layer::getOutput()
{
	return this->m_output;
}

// Get the neuron with the given index from the layer
Neuron * Layer::getNeuron(int index)
{
	return this->m_neurons[index];
}

// Compute the output of this layer with a given input
double * Layer::compute(double * inputs)
{
	// Calculate the output for each neuron
	for(int i = 0; i < this->m_neuronCount; i++)
		this->m_output[i] = this->m_neurons[i]->compute(inputs);

	// Return the new output
	return this->m_output;
}

// Randomize the weights of the neurons
void Layer::randomize()
{
	for(int i = 0; i < this->m_neuronCount; i++)
		this->m_neurons[i]->randomize();
}