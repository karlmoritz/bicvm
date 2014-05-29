// File: openqa_bordes_trainer.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 16-01-2013
// Last Update: Thu 29 May 2014 14:17:54 BST

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
      Real *gradient_location,
      int n,
      int iteration,
      BProps& prop,
      Real* error,
      bool near_noise);

  void testModel(Model &model);

  void setVarsAndNumber(Real *&vars, int &number_vars, Model &model);

};

#endif  // COMMON_OPENQA_BORDES_TRAINER_H
