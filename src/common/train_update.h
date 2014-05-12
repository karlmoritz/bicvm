// File: train_update.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 16-01-2013
// Last Update: Thu 02 Jan 2014 03:37:01 PM GMT

#ifndef COMMON_TRAIN_UPDATE_H
#define COMMON_TRAIN_UPDATE_H

// L-BFGS
#include <lbfgs.h>

// Local
#include "shared_defs.h"

void computeCostAndGrad(
    Model &model,
    const Real *x, // Variables (theta)
    Real *g,       // Put gradient here
    int n,                   // Number of variables
    int iteration, // Current iteration
    BProps& prop,
    Real* error);

void computeBiCostAndGrad(
    Model &modelA,
    Model &modelB,
    const Real *x,
    Real *gradient_location,
    int n,
    int iteration,
    BProps& prop,
    Real* error);

void testModel(Model &model);

void setVarsAndNumber(Real *&vars, int &number_vars, Model &model);

#endif  // COMMON_TRAIN_UPDATE_H
