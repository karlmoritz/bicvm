// File: recursive_autoencoder.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 02-01-2013
// Last Update: Tue 13 May 2014 15:10:45 BST

#include "recursive_autoencoder.h"

#include "backpropagator.h"
#include "singleprop.h"

#include <iostream>

// Namespaces
using namespace std;

namespace additive {

RecursiveAutoencoder::RecursiveAutoencoder(const ModelData& config) :
  RecursiveAutoencoderBase(config), Theta_Wl(0,0), Theta_Bl(0,0) {}

RecursiveAutoencoder::~RecursiveAutoencoder() {}

RecursiveAutoencoderBase* RecursiveAutoencoder::cloneEmpty()  {
  return new RecursiveAutoencoder(config);
}

BackpropagatorBase* RecursiveAutoencoder::getBackpropagator(const Model &model,
                                                            int n)
{
  BackpropagatorBase* bpb = new Backpropagator(this, model, n);
  return bpb;
}
SinglePropBase* RecursiveAutoencoder::getSingleProp(int sl, int nl, Real beta, Bools updates)
{
  SinglePropBase* spb = new SingleProp(this, sl, nl, beta, updates);
  return spb;
}

void RecursiveAutoencoder::init(bool init_words, bool create_new_theta) {
  int word_width = config.word_representation_size;
  int label_width = config.label_class_size;

  theta_Wl_size_  = word_width * label_width;
  theta_Bl_size_  = label_width;
  theta_size_ = theta_Wl_size_ + theta_Bl_size_;

  if (create_new_theta)
    theta_ = new Real[theta_size_];

  theta_Wl_  = theta_;
  theta_Bl_  = theta_Wl_ + theta_Wl_size_;

  Real* ptr = theta_;
  new (&Theta_Full) WeightVectorType(ptr, theta_size_);
  new (&Theta_Wl) WeightVectorType(ptr, theta_Wl_size_); ptr += theta_Wl_size_;
  new (&Theta_Bl) WeightVectorType(ptr, theta_Bl_size_); ptr += theta_Bl_size_;
  assert (ptr == theta_ + theta_size_);

  //std::random_device rd;
  //std::mt19937 gen(rd());
  std::mt19937 gen(0);
  Real r1 = 6.0 / sqrt( 2 * word_width );
  Real r2 = 1.0 / sqrt( word_width );
  std::uniform_real_distribution<> dis1(-r1,r1);
  std::uniform_real_distribution<> dis2(-r2,r2);
  std::uniform_real_distribution<> d_sud(0,1);    // Matlab rand / standard uniform distribution
  std::normal_distribution<> d_snd(0,1);          // Matlab randn / standard normal distribution
  ptr = theta_;

  // Initialize Label Matrix and Weight
  Wl.clear();
  Wl.push_back(WeightMatrixType(ptr, label_width, word_width));
  ptr += label_width*word_width;
  if (init_words) {
    for (int i = 0; i < theta_Wl_size_; ++i) Theta_Wl(i) = dis1(gen);
  }

  // Initialize Label Bias and Weight
  Bl.clear();
  Bl.push_back(WeightVectorType(ptr, label_width));
  ptr += label_width;
  if (init_words) {
    Bl.back().setZero();
  }
  assert(ptr == theta_+theta_size_);
}


Real RecursiveAutoencoder::getLambdaCost(Bools l, Lambdas lambdas)
{
  Real lcost = 0.0;
  if (l.Wl)  lcost += lambdas.alpha_lbl * lambdas.Wl  * 0.5 * Theta_Wl.cwiseProduct(Theta_Wl).sum();
  if (l.Bl)  lcost += lambdas.alpha_lbl * lambdas.Bl  * 0.5 * Theta_Bl.cwiseProduct(Theta_Bl).sum();
  return lcost;
}

void RecursiveAutoencoder::addLambdaGrad(Real* theta_data, Bools l, Lambdas lambdas)
{
  theta_data += theta_D_size_;
  if (l.Wl)
  {
    WeightVectorType X = WeightVectorType(theta_data,theta_Wl_size_);
    X += (Theta_Wl * lambdas.alpha_lbl * lambdas.Wl);
  }
  theta_data += theta_Wl_size_;
  if (l.Bl)
  {
    WeightVectorType X = WeightVectorType(theta_data,theta_Bl_size_);
    X += (Theta_Bl * lambdas.alpha_lbl * lambdas.Bl);
  }
  theta_data += theta_Bl_size_;
}

void RecursiveAutoencoder::setIncrementalCounts(Counts *counts, Real *&vars, int &number)
{
  vars = theta_;
  counts->D   = theta_D_size_;
  counts->Wl  = counts->D + theta_Wl_size_;
  counts->Bl  = counts->Wl  + theta_Bl_size_;
  number = theta_size_;
}

}  // namespace additive
