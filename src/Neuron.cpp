//  _    _ _      _ _          _    _ _      _ _                        
// | |  | (_)    | | |        | |  | (_)    | | |                       
// | |__| |_  ___| | | _____  | |__| |_  ___| | | _____ _ __ ___   __ _ 
// |  __  | |/ _ \ | |/ / _ \ |  __  | |/ _ \ | |/ / _ \ '_ ` _ \ / _` |
// | |  | | |  __/ |   <  __/ | |  | | |  __/ |   <  __/ | | | | | (_| |
// |_|  |_|_|\___|_|_|\_\___| |_|  |_|_|\___|_|_|\_\___|_| |_| |_|\__,_|
// _____________________________________________________________________
//
// Filename: Neuron.cpp
// Project: AI - Neural network
// Created: 26-06-2014 (DD-MM-YYYY)
// Changed: 27-06-2014 (DD-MM-YYYY)
//
// Author: Hielke Hielkema
// Contact: HielkeHielkema93@gmail.com
//
// (C) Hielke Hielkema - 2014

#include "Neuron.h"

// Constructor
Neuron::Neuron(int inputs, ActivationFunction * function)
{
	// Minimum of 1
	if (inputs < 1)
		inputs = 1;

	// Initialize data
	this->m_inputsCount = inputs;
	this->m_weights = new double[inputs];
	this->m_output = 0;
	this->m_threshold = 0.0f;
	this->m_function = function;

	// Randomize the weights
	this->randomize();
}

// Destructor
Neuron::~Neuron(void)
{
	// Delete resources
	delete[] this->m_weights;
}

// Get the number of inputs
int Neuron::getInputsCount()
{
	return this->m_inputsCount;
}

// Get the output
double Neuron::getOutput()
{
	return this->m_output;
}

// Get the threshold
double Neuron::getThreshold()
{
	return this->m_threshold;
}

// Increase the threshold
void Neuron::increaseThreshold(double value)
{
	this->m_threshold += value;
}

// Get the activation function
ActivationFunction * Neuron::getActivationFunction()
{
	return this->m_function;
}

// Get the weight of one of the inputs
double Neuron::getWeight(int index)
{
	return this->m_weights[index];
}

// Set the weight of one of the inputs
void Neuron::setWeight(int index, double value)
{
	this->m_weights[index] = value;
}

// Increase the weight of one of the inputs
void Neuron::increaseWeight(int index, double value)
{
	this->m_weights[index] += value;
}

// Randomize the weights
void Neuron::randomize()
{
	for(int i = 0; i < this->m_inputsCount; i++) 
		this->m_weights[i] = this->getRandom();
}

// Compute the output
double Neuron::compute(double * inputs)
{
	// Initalize the sum value
	double sum = 0.0;

	// Calculate the sum of al inputs multiplied with there weight
	for(int i = 0; i < this->m_inputsCount; i++) 
		sum += this->m_weights[i] * inputs[i];

	// Add the threshold
	sum += this->m_threshold;

	// Apply the function
	this->m_output = this->m_function->function(sum);

	// Return the output
	return this->m_output;
}

// Get a random number between 0.0 and 1.0
double Neuron::getRandom()
{
	return (double)rand() / RAND_MAX;
}