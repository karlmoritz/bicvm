// File: train_sgd.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 09-01-2013
// Last Update: Thu 17 Oct 2013 04:21:03 PM BST

#ifndef COMMON_TRAIN_SGD_H
#define COMMON_TRAIN_SGD_H

// Local
#include "shared_defs.h"

int train_sgd(Model &model, int iterations, Real eta, Lambdas lambdas);

#endif  // COMMON_TRAIN_SGD_H

