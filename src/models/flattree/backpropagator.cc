// File: backpropagator.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 22-04-2013
// Last Update: Fri 30 May 2014 15:11:23 BST

#include "models/flattree/backpropagator.h"

#include "models/flattree/recursive_autoencoder.h"
#include "models/flattree/singleprop.h"

namespace flattree {

Backpropagator::Backpropagator(RecursiveAutoencoder* rae, const Model &model,
                               int n, Real* dictptr)
  : BackpropagatorBase(rae, model),
  grad_D(0, 0, 0), grad_Wl(0, 0), grad_Bl(0, 0) {

    if (dictptr != nullptr) { assert (n == rae->getThetaSize()); }

    data = new Real[n];
    Real *ptr = data;

    if (dictptr == nullptr) {
      new (&grad_D) WeightMatrixType(ptr, dict_size, word_width);
      grad_D.setZero();
      ptr += rae->getThetaDSize();
    }
    new (&grad_Wl) WeightVectorType(ptr, rae->theta_Wl_size_);
    grad_Wl.setZero();
    ptr += rae->theta_Wl_size_;
    new (&grad_Bl) WeightVectorType(ptr, rae->theta_Bl_size_);
    grad_Bl.setZero();
    ptr += rae->theta_Bl_size_;

    assert (data + n == ptr);

    if (dictptr != nullptr) {
      singleprop->passDictLink(dictptr, rae->getThetaDSize());
      singleprop->passDataLink(data, rae->getThetaSize());
      new (&dict_weights) WeightVectorType(dictptr, rae->getThetaDSize());
      new (&weights) WeightVectorType(data, rae->getThetaSize());
    } else {
      singleprop->passDictLink(data, rae->getThetaDSize());
      singleprop->passDataLink(data+rae->getThetaDSize(), rae->getThetaSize());
      new (&dict_weights) WeightVectorType(data, rae->getThetaDSize());
      new (&weights) WeightVectorType(data+rae->getThetaDSize(), rae->getThetaSize());
    }
  }

Backpropagator::~Backpropagator() {
  delete [] data;
}

void Backpropagator::collectGradients(SinglePropBase* spb_, int i) {
  /*
   * {
   *   auto tmpD = spb->getDGradients();
   *   for (size_t k = 0; k < model.corpus[i].words.size(); ++k)
   *     grad_D.row(model.corpus[i].words[k]) += tmpD[k];
   * }
   */
}

void Backpropagator::normalize(int type) { // type: 0=rae, 1=lbl, 2=bi
}

}  // namespace flattree
