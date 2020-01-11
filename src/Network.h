//  _    _ _      _ _          _    _ _      _ _                        
// | |  | (_)    | | |        | |  | (_)    | | |                       
// | |__| |_  ___| | | _____  | |__| |_  ___| | | _____ _ __ ___   __ _ 
// |  __  | |/ _ \ | |/ / _ \ |  __  | |/ _ \ | |/ / _ \ '_ ` _ \ / _` |
// | |  | | |  __/ |   <  __/ | |  | | |  __/ |   <  __/ | | | | | (_| |
// |_|  |_|_|\___|_|_|\_\___| |_|  |_|_|\___|_|_|\_\___|_| |_| |_|\__,_|
// _____________________________________________________________________
//
// Filename: Network.h
// Project: AI - Neural network
// Created: 26-06-2014 (DD-MM-YYYY)
// Changed: 27-06-2014 (DD-MM-YYYY)
//
// Author: Hielke Hielkema
// Contact: HielkeHielkema93@gmail.com
//
// (C) Hielke Hielkema - 2014

#include "Layer.h";

#pragma once
class Network
{
public:
	Network(ActivationFunction * function, int inputsCount, int layerCount, int * neuronCounts);
	~Network(void);

	int getInputsCount();
	int getLayersCount();
	double * getOutput();
	Layer * getLayer(int index);
	double * compute(double * input);
	void randomize();
	ActivationFunction * getFunction();

private:
	int m_inputCount;
	int m_layerCount;
	double * m_output;
	Layer ** m_layers;
};