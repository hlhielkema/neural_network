//  _    _ _      _ _          _    _ _      _ _                        
// | |  | (_)    | | |        | |  | (_)    | | |                       
// | |__| |_  ___| | | _____  | |__| |_  ___| | | _____ _ __ ___   __ _ 
// |  __  | |/ _ \ | |/ / _ \ |  __  | |/ _ \ | |/ / _ \ '_ ` _ \ / _` |
// | |  | | |  __/ |   <  __/ | |  | | |  __/ |   <  __/ | | | | | (_| |
// |_|  |_|_|\___|_|_|\_\___| |_|  |_|_|\___|_|_|\_\___|_| |_| |_|\__,_|
// _____________________________________________________________________
//
// Filename: NeuralNetworkCpp.cpp
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
#include "SigmoidFunction.h"
#include "BackPropagationLearning.h"

int _tmain(int argc, _TCHAR* argv[])
{
	// Set random seed
	srand (time(NULL));

	// Network config
	int inputCount = 2;
	int layerCount = 2;
	int neuronCounts[] = { 2, 1 } ;

	// Initialize network and learning
	ActivationFunction * function = new SigmoidFunction();
	Network * network = new Network(function, inputCount, layerCount, neuronCounts);
	BackPropagationLearning * teacher = new BackPropagationLearning(network);

	// Create the learning set
	struct Combination
	{
		double a;
		double b;
		double out;
	};
	const int rounds = 1000000; // learning cycles
	const int setSize = 4;
	const int outputCount = 1;
	Combination set[setSize] = {
		{ 1, 1, 0 }, // 1 XOR 1 = 0
		{ 1, 0, 1 }, // 1 XOR 0 = 1
		{ 0, 1, 1 }, // 0 XOR 1 = 1
		{ 0, 0, 0 }  // 0 XOR 0 = 0
	};

	double ** input = new double*[setSize];
	double ** output = new double*[setSize];
	for(int i = 0; i < setSize; i++)
	{
		input[i] = new double[inputCount];
		output[i] = new double[outputCount];

		input[i][0] = set[i].a;
		input[i][1] = set[i].b;
		output[i][0] = set[i].out;
	}

	// Do the learning cycles
	for(int p = 1; p <= 100; p++)
	{
		for(int r = 0; r < rounds / 100; r++)
			teacher->runEpoch(input, setSize, output);
		std::cout << p << '%' << std::endl;
	}

	// Print the end result
	std::cout << "XOR results:" << std::endl;
	for(int i = 0; i < setSize; i++)
	{
		double input_arr[] =  { set[i].a, set[i].b };
		double * results = network->compute(input_arr);
		double result = results[0];
		std::cout << set[i].a << " XOR " << set[i].b << " = " << result << std::endl;
	}

	// Delete resources
	delete function;
	delete network;
	delete teacher;

	int pause;
	std::cin >> pause;

	return 0;
}

