// File: trainer.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 16-01-2013
// Last Update: Mon 12 May 2014 17:55:58 BST

#ifndef COMMON_TRAINER_H
#define COMMON_TRAINER_H

// L-BFGS
#include <lbfgs.h>

// Local
#include "shared_defs.h"

class Trainer {
 public:
  virtual void computeCostAndGrad(
      Model &model,
      const Real *x, // Variables (theta)
      Real *g,       // Put gradient here
      int n,                   // Number of variables
      int iteration, // Current iteration
      BProps& prop,
      Real* error) = 0;

  virtual void testModel(Model &model) = 0;
  virtual void setVarsAndNumber(Real *&vars, int &number_vars, Model &model) = 0;
};

#endif  // COMMON_TRAINER_H
