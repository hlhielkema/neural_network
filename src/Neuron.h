//  _    _ _      _ _          _    _ _      _ _                        
// | |  | (_)    | | |        | |  | (_)    | | |                       
// | |__| |_  ___| | | _____  | |__| |_  ___| | | _____ _ __ ___   __ _ 
// |  __  | |/ _ \ | |/ / _ \ |  __  | |/ _ \ | |/ / _ \ '_ ` _ \ / _` |
// | |  | | |  __/ |   <  __/ | |  | | |  __/ |   <  __/ | | | | | (_| |
// |_|  |_|_|\___|_|_|\_\___| |_|  |_|_|\___|_|_|\_\___|_| |_| |_|\__,_|
// _____________________________________________________________________
//
// Filename: Neuron.h
// Project: AI - Neural network
// Created: 26-06-2014 (DD-MM-YYYY)
// Changed: 27-06-2014 (DD-MM-YYYY)
//
// Author: Hielke Hielkema
// Contact: HielkeHielkema93@gmail.com
//
// (C) Hielke Hielkema - 2014

#include "stdafx.h"
#include "ActivationFunction.h"

#pragma once
class Neuron
{
public:
	Neuron(int inputs, ActivationFunction * function);
	~Neuron(void);

	int getInputsCount();
	double getOutput();
	double getThreshold();
	void increaseThreshold(double value);
	ActivationFunction * getActivationFunction();
	double getWeight(int index);
	void setWeight(int index, double value);
	void increaseWeight(int index, double value);
	void randomize();
	double compute(double * input);

private:
	int	m_inputsCount;
	double * m_weights;
	double m_output;
    double m_threshold;
    ActivationFunction * m_function;

	double getRandom();
};