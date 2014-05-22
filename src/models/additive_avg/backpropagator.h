// File: backpropagator.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 22-04-2013
// Last Update: Thu 22 May 2014 11:44:28 BST

#ifndef MODELS_ADDITIVE_AVG_BACKPROPAGATOR_H
#define MODELS_ADDITIVE_AVG_BACKPROPAGATOR_H

// Local
#include "../../common/shared_defs.h"
#include "../../common/backpropagatorbase.h"

class SinglePropBase;

namespace additive_avg {

class RecursiveAutoencoder;

class Backpropagator : public BackpropagatorBase {
 public:
  Backpropagator (RecursiveAutoencoder* rae, const Model &model, int n);
  ~Backpropagator ();

  /*virtual*/void collectGradients(SinglePropBase* spb, int sentence);
  /*virtual*/void normalize(int type);

 private:
  WeightMatrixType grad_D;
  WeightVectorType grad_Wl;
  WeightVectorType grad_Bl;

  Real* data;
};

}  // namespace additive_avg
#endif  // MODELS_ADDITIVE_AVG_BACKPROPAGATOR_H
