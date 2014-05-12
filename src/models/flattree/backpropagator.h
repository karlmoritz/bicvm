// File: backpropagator.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 22-04-2013
// Last Update: Fri 18 Oct 2013 02:59:47 PM BST

#ifndef MODELS_FLATTREE_BACKPROPAGATOR_H
#define MODELS_FLATTREE_BACKPROPAGATOR_H

// L-BFGS
#include <lbfgs.h>

// Local
#include "../../common/shared_defs.h"
#include "../../common/backpropagatorbase.h"

class SinglePropBase;

namespace flattree {

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

}  // namespace flattree
#endif  // MODELS_FLATTREE_BACKPROPAGATOR_H
