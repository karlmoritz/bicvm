// File: recursive_autoencoder.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 02-01-2013
// Last Update: Fri 18 Oct 2013 02:11:24 PM BST

#ifndef MODELS_ADDITIVE_RECURSIVE_AUTOENCODER_H
#define MODELS_ADDITIVE_RECURSIVE_AUTOENCODER_H

#include "../../common/recursive_autoencoder.h"
#include <lbfgs.h>

namespace additive {

class Backpropagator;
class SingleProp;

class RecursiveAutoencoder : public RecursiveAutoencoderBase {
 public:
  RecursiveAutoencoder (const ModelData& config);

  virtual ~RecursiveAutoencoder ();
  RecursiveAutoencoderBase* cloneEmpty ();

  Real getLambdaCost(Bools bl, Lambdas lambdas);
  void addLambdaGrad(Real* theta_data, Bools bl, Lambdas lambdas);

  void setIncrementalCounts(Counts *counts, Real *&vars, int &number);

  BackpropagatorBase* getBackpropagator(const Model &model, int n);
  SinglePropBase*     getSingleProp(int sl, int nl, Real beta, Bools updates);

  friend class SingleProp;
  friend class Backpropagator;
 private:
  void init(bool init_words, bool create_new_theta=true);

  WeightMatricesType    Wl;     // Label (nxl)
  WeightVectorsType     Bl;     // Bias  (l)

  Real*                 theta_Wl_;
  Real*                 theta_Bl_;

  int         theta_Wl_size_;
  int         theta_Bl_size_;

  WeightVectorType      Theta_Wl;
  WeightVectorType      Theta_Bl;
};

}  // namespace additive

#endif  // MODELS_ADDITIVE_RECURSIVE_AUTOENCODER_H
