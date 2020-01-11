//  _    _ _      _ _          _    _ _      _ _                        
// | |  | (_)    | | |        | |  | (_)    | | |                       
// | |__| |_  ___| | | _____  | |__| |_  ___| | | _____ _ __ ___   __ _ 
// |  __  | |/ _ \ | |/ / _ \ |  __  | |/ _ \ | |/ / _ \ '_ ` _ \ / _` |
// | |  | | |  __/ |   <  __/ | |  | | |  __/ |   <  __/ | | | | | (_| |
// |_|  |_|_|\___|_|_|\_\___| |_|  |_|_|\___|_|_|\_\___|_| |_| |_|\__,_|
// _____________________________________________________________________
//
// Filename: BackPropagationLearning.h
// Project: AI - Neural network
// Created: 26-06-2014 (DD-MM-YYYY)
// Changed: 27-06-2014 (DD-MM-YYYY)
//
// Author: Hielke Hielkema
// Contact: HielkeHielkema93@gmail.com
//
// (C) Hielke Hielkema - 2014

#include "Network.h"
#include "stdafx.h"

#pragma once
class BackPropagationLearning
{
public:
	BackPropagationLearning(Network * network);
	~BackPropagationLearning(void);

	double getLearningRate();
	void setLearningRate(double learningRate);
	double getMomentum();
	void setMomentum(double momentum);

	double run(double * input, double * output);
	double runEpoch(double ** input, int setSize, double ** output);
	double calulcateError(double * desiredOutput);
	void calulcateUpdates(double * input);
	void updateNetwork();

private:
	Network * m_network;
	double m_learningRate;
	double m_momentum;

	double ** m_neuronErrors;
	double *** m_weightUpdates;
	double ** m_thresholdUpdates;
};

