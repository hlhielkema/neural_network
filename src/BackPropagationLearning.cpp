//  _    _ _      _ _          _    _ _      _ _                        
// | |  | (_)    | | |        | |  | (_)    | | |                       
// | |__| |_  ___| | | _____  | |__| |_  ___| | | _____ _ __ ___   __ _ 
// |  __  | |/ _ \ | |/ / _ \ |  __  | |/ _ \ | |/ / _ \ '_ ` _ \ / _` |
// | |  | | |  __/ |   <  __/ | |  | | |  __/ |   <  __/ | | | | | (_| |
// |_|  |_|_|\___|_|_|\_\___| |_|  |_|_|\___|_|_|\_\___|_| |_| |_|\__,_|
// _____________________________________________________________________
//
// Filename: BackPropagationLearning.cpp
// Project: AI - Neural network
// Created: 26-06-2014 (DD-MM-YYYY)
// Changed: 27-06-2014 (DD-MM-YYYY)
//
// Author: Hielke Hielkema
// Contact: HielkeHielkema93@gmail.com
//
// (C) Hielke Hielkema - 2014

#include "BackPropagationLearning.h"

// Constructor
BackPropagationLearning::BackPropagationLearning(Network * network)
{
	// Initialize members
	this->m_network = network;
	this->m_learningRate = 0.1;
	this->m_momentum = 0.0;

	// Get the layer count of the network
	int layerCount = network->getLayersCount();

	// Initialize 2/3-dim arrays
	this->m_neuronErrors = new double*[layerCount];
	this->m_weightUpdates = new double**[layerCount];
	this->m_thresholdUpdates = new double*[layerCount];

	// Update errors and updates arrays
	for(int i = 0; i < layerCount; i++)
	{
		// Get the neuron count of the layer
		int neuronCount = network->getLayer(i)->getNeuronsCount();

		// Alloc the second dimmension of the arrays
		this->m_neuronErrors[i] = new double[neuronCount];
		this->m_weightUpdates[i] = new double*[neuronCount];
		this->m_thresholdUpdates[i] = new double[neuronCount];

		// for each neuron
		for (int j = 0; j < neuronCount; j++)
		{
			// Create the array of weight updates fir each input
			this->m_weightUpdates[i][j] = new double[network->getLayer(i)->getInputsCount()];
		}
	}
}

// Descructor
BackPropagationLearning::~BackPropagationLearning(void)
{
	// Delete the content of the errors and updates arrays
	for(int i = 0; i < this->m_network->getLayersCount(); i++)
	{
		// Get the neuron count of the layer
		int neuronCount = this->m_network->getLayer(i)->getNeuronsCount();

		// Delete the weights of each neuron
		for (int j = 0; j < neuronCount; j++)
		{
			delete[] this->m_weightUpdates[i][j];
		}

		// Delete the second dimmension of the array
		delete[] this->m_neuronErrors[i];
		delete[] this->m_weightUpdates[i];
		delete[] this->m_thresholdUpdates[i];
	}

	// Delete arrays
	delete[] this->m_neuronErrors;
	delete[] this->m_weightUpdates;
	delete[] this->m_thresholdUpdates;
}

// Get the learning rate
double BackPropagationLearning::getLearningRate()
{
	return this->m_learningRate;
}

// Set the learning rate
void BackPropagationLearning::setLearningRate(double learningRate)
{
	this->m_learningRate = learningRate;
}

// Get the momentum
double BackPropagationLearning::getMomentum()
{
	return this->m_momentum;
}

// Set the momentum
void BackPropagationLearning::setMomentum(double momentum)
{
	this->m_momentum = momentum;
}

// Run a learning iteration
// @Returns sum of squared errors of the last layer divided by 2
double BackPropagationLearning::run(double * input, double * output)
{
	// Compute the network's output
	this->m_network->compute(input);

    // Calculate network error
	double error = this->calulcateError(output);

	// Calculate the weight and threshold updates
	this->calulcateUpdates(input);

	// Update the network
	this->updateNetwork();

	// Return the update
    return error;
}

// Run a learning epoch
// @Returns sum of squared errors of the last layer divided by 2
double BackPropagationLearning::runEpoch(double ** input, int setSize, double ** output)
{
	double error = 0.0;
	
	// run learning procedure for all samples
	for (int i = 0; i < setSize; i++)
		error += this->run(input[i], output[i]);

	// return the error
	return error;
}

// Calculates error values for all neurons of the network
// Note: assumes that all neurons of the network have the same activation function
// @Returns sum of squared errors of the last layer divided by 2
double BackPropagationLearning::calulcateError(double * desiredOutput)
{
	// Current and next layer
	Layer * layer, * layerNext;
	// Current and next error arrays
	double * errors, * errorsNext;
	// Error values
	double error = 0, e, sum;
	// Neuron output value
	double output;

	// Get the layer count
	int layerCount = this->m_network->getLayersCount();

	// Get the activation function (assume, that all neurons of the network have the same activation function)
	ActivationFunction * function = this->m_network->getFunction();

	// Calculate error values for the last layer first
	layer = this->m_network->getLayer(layerCount - 1);
	errors = this->m_neuronErrors[layerCount - 1];
	for (int i = 0, n = layer->getNeuronsCount(); i < n; i++)
    {
		// Get the output of the neuron
		output = layer->getNeuron(i)->getOutput();

		// Error of the neuron
		e = desiredOutput[i] - output;

		// Error multiplied with activation function's derivative
		errors[i] = e * function->derivative2(output);

		// Square the error and sum it
        error += (e * e);
    }

	// Calculate error values for other layers
    for (int j = layerCount - 2; j >= 0; j--)
    {
		// Get next layer
		layer = this->m_network->getLayer(j);
		layerNext = this->m_network->getLayer(j + 1);
		errors = this->m_neuronErrors[j];
		errorsNext = this->m_neuronErrors[j + 1];

        // For all neurons of the layer
		for (int i = 0, n = layer->getNeuronsCount(); i < n; i++)
        {
            sum = 0.0;
            // For all neurons of the next layer
			for (int k = 0, m = layerNext->getNeuronsCount(); k < m; k++)
				sum += errorsNext[k] * layerNext->getNeuron(k)->getWeight(i);
            errors[i] = sum * function->derivative2(layer->getNeuron(i)->getOutput());
        }
    }

    // Return squared error of the last layer divided by 2
    return error / 2.0;
}

// Calculate the weight and threshold updates
void BackPropagationLearning::calulcateUpdates(double * input)
{
	// Current neuron
	Neuron * neuron;
    // Current and previous layers
    Layer * layer, * layerPrev;
    // Layer's weights updates
    double ** layerWeightsUpdates;
    // Layer's thresholds updates
    double * layerThresholdUpdates;
    // Layer's error
    double * errors;
    // Neuron's weights updates
    double * neuronWeightUpdates;
    // Error value
    double error;

	// calculate updates for the last layer first
	layer = this->m_network->getLayer(0);
	errors = this->m_neuronErrors[0];
	layerWeightsUpdates = this->m_weightUpdates[0];
	layerThresholdUpdates = this->m_thresholdUpdates[0];

	// For each neuron of the layer
	for (int i = 0, n = this->m_network->getLayersCount(); i < n; i++)
    {
		neuron = layer->getNeuron(i);
        error = errors[i];
        neuronWeightUpdates = layerWeightsUpdates[i];
		
        // for each weight of the neuron
		for (int j = 0, m = neuron->getInputsCount(); j < m; j++)
        {
            // calculate weight update
			neuronWeightUpdates[j] = this->m_learningRate * (this->m_momentum * neuronWeightUpdates[j] + (1.0 - this->m_momentum) * error * input[j]);
        }

        // calculate treshold update
        layerThresholdUpdates[i] = this->m_learningRate * (this->m_momentum * layerThresholdUpdates[i] + (1.0 - this->m_momentum) * error);
    }

    // for all other layers
	for (int k = 1, l = this->m_network->getLayersCount(); k < l; k++)
    {
		layerPrev = this->m_network->getLayer(k - 1);
        layer = this->m_network->getLayer(k);
		errors = this->m_neuronErrors[k];
		layerWeightsUpdates = this->m_weightUpdates[k];
		layerThresholdUpdates = this->m_thresholdUpdates[k];
		
		
        // for each neuron of the layer
		for (int i = 0, n = layer->getNeuronsCount(); i < n; i++)
        {
			neuron = layer->getNeuron(i);
            error = errors[i];
            neuronWeightUpdates = layerWeightsUpdates[i];
			
            // for each synapse of the neuron
			for (int j = 0, m = neuron->getInputsCount(); j < m; j++)
            {
                // calculate weight update
				neuronWeightUpdates[j] = this->m_learningRate * (this->m_momentum * neuronWeightUpdates[j] + (1.0 - this->m_momentum) * error * layerPrev->getNeuron(j)->getOutput());
            }

            // calculate treshold update
            layerThresholdUpdates[i] = this->m_learningRate * (this->m_momentum * layerThresholdUpdates[i] + (1.0 - this->m_momentum) * error);
				
        }
    }
}

// Update the weights and thresholds in the network with the calculated values
void BackPropagationLearning::updateNetwork()
{
	// Current neuron
	Neuron * neuron;
	// Current layer
	Layer * layer;
    // Layer's weights updates
    double ** layerWeightsUpdates;
    // Layer's thresholds updates
    double * layerThresholdUpdates;
    // Neuron's weights updates
    double * neuronWeightUpdates;

	// for each layer of the network
	for (int i = 0, n = this->m_network->getLayersCount(); i < n; i++)
    {
		layer = this->m_network->getLayer(i);
		layerWeightsUpdates = this->m_weightUpdates[i];
		layerThresholdUpdates = this->m_thresholdUpdates[i];

        // for each neuron of the layer
		for (int j = 0, m = layer->getNeuronsCount(); j < m; j++)
        {
			neuron = layer->getNeuron(j);
            neuronWeightUpdates = layerWeightsUpdates[j];

            // for each weight of the neuron
			for (int k = 0, s = neuron->getInputsCount(); k < s; k++)
            {
                // update weight
				neuron->increaseWeight(k, neuronWeightUpdates[k]);
            }
            // update treshold
			neuron->increaseThreshold(layerThresholdUpdates[j]);
        }
    }
}
