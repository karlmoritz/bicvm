// File: backpropagator.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 22-04-2013
// Last Update: Wed 14 May 2014 12:40:46 BST

#include "models/additive/backpropagator.h"

#include "models/additive/recursive_autoencoder.h"
#include "models/additive/singleprop.h"

namespace additive {

Backpropagator::Backpropagator(RecursiveAutoencoder* rae, const Model &model,
                               int n)
  : BackpropagatorBase(rae, model),
  grad_D(0, 0, 0), grad_Wl(0, 0), grad_Bl(0, 0) {

    data = new Real[n];
    Real *ptr = data;

    new (&grad_D) WeightMatrixType(ptr, dict_size, word_width);
    grad_D.setZero();
    ptr += rae->getThetaDSize();
    new (&grad_Wl) WeightVectorType(ptr, rae->theta_Wl_size_);
    grad_Wl.setZero();
    ptr += rae->theta_Wl_size_;
    new (&grad_Bl) WeightVectorType(ptr, rae->theta_Bl_size_);
    grad_Bl.setZero();
    ptr += rae->theta_Bl_size_;

    assert (data + n == ptr);

    singleprop->passDataLink(data, n);
    new (&dict_weights) WeightVectorType(data,rae->getThetaDSize());
    new (&weights) WeightVectorType(data+rae->getThetaDSize(),rae->getThetaSize());
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

}  // namespace additive
