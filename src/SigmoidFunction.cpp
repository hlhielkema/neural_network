//  _    _ _      _ _          _    _ _      _ _                        
// | |  | (_)    | | |        | |  | (_)    | | |                       
// | |__| |_  ___| | | _____  | |__| |_  ___| | | _____ _ __ ___   __ _ 
// |  __  | |/ _ \ | |/ / _ \ |  __  | |/ _ \ | |/ / _ \ '_ ` _ \ / _` |
// | |  | | |  __/ |   <  __/ | |  | | |  __/ |   <  __/ | | | | | (_| |
// |_|  |_|_|\___|_|_|\_\___| |_|  |_|_|\___|_|_|\_\___|_| |_| |_|\__,_|
// _____________________________________________________________________
//
// Filename: SigmoidFunction.cpp
// Project: AI - Neural network
// Created: 26-06-2014 (DD-MM-YYYY)
// Changed: 27-06-2014 (DD-MM-YYYY)
//
// Author: Hielke Hielkema
// Contact: HielkeHielkema93@gmail.com
//
// (C) Hielke Hielkema - 2014

#include "SigmoidFunction.h"
#include "stdafx.h"

SigmoidFunction::SigmoidFunction()
{
	this->m_alpha = 2;
}

SigmoidFunction::SigmoidFunction(double alpha)
{
	this->m_alpha = alpha;
}

double SigmoidFunction::function(double x)
{
	return 1 / (1 + exp(-this->m_alpha * x));
}

//  Calculates function derivative
double SigmoidFunction::derivative(double x)
{
	double y = this->function(x);
	return (this->m_alpha * y * (1 - y));
}

// Calculates function derivative
double SigmoidFunction::derivative2(double y)
{
	return (this->m_alpha * y * (1 - y));
}