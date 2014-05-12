// File: train_adagrad.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 09-01-2013
// Last Update: Fri 11 Oct 2013 03:53:14 PM BST

#ifndef COMMON_TRAIN_ADAGRAD_H
#define COMMON_TRAIN_ADAGRAD_H

// Local
#include "shared_defs.h"

int train_adagrad(Model &model, int iterations, Real eta, Model *tmodel, int
                  batches, Lambdas lambdas, Real l1);

#endif  // COMMON_TRAIN_ADAGRAD_H

