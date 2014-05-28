// File: openqa_bordes_trainer.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 16-01-2013
// Last Update: Wed 28 May 2014 19:09:58 BST

#ifndef COMMON_OPENQA_BORDES_TRAINER_H
#define COMMON_OPENQA_BORDES_TRAINER_H

// Local
#include "shared_defs.h"
#include "trainer.h"

class OpenQABordesTrainer : public Trainer {
 public:

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

};

#endif  // COMMON_OPENQA_BORDES_TRAINER_H
