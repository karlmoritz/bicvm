// File: recursive_autoencoder.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 02-01-2013
// Last Update: Mon 06 Jan 2014 09:46:09 PM GMT

#include "recursive_autoencoder.h"

#include <iostream>

// Namespaces
using namespace std;

RecursiveAutoencoderBase::RecursiveAutoencoderBase(const ModelData& config) :
  config(config), D(0,0,0), Theta_Full(0,0), Theta_D(0,0)
{ theta_ = nullptr; }


RecursiveAutoencoderBase::~RecursiveAutoencoderBase () {
  delete [] theta_;
}

void RecursiveAutoencoderBase::finalizeDictionary(bool random_init)
{
  delete [] theta_;
  init(random_init,true);
}

void RecursiveAutoencoderBase::finalizeSpecific(Real* theta_address)
{
  for (int i=0; i<theta_size_; ++i) {
    theta_address[i] = theta_[i];
    theta_[i] = 0.5;
  }
  delete [] theta_;
  theta_ = theta_address;
  cout << "Fint " << "theta_ " << theta_ << "   : " << theta_[0] << "  " << &theta_[0] << endl;
  init(false,false);
}

void RecursiveAutoencoderBase::debugSize(int count)
{
  std::cout << "theta " << count << ": " << Theta_Full.sum() << endl;
}

void RecursiveAutoencoderBase::initFromWithDict(
    const RecursiveAutoencoderBase& rae,
    std::map<LabelID, LabelID> n2o_map) {
  for (auto i = 0; i < (rae.theta_size_ - rae.theta_D_size_); ++i)
    theta_[theta_D_size_ + i] = rae.theta_[rae.theta_D_size_ + i];
  // Update WE using the n2o_map
  for (auto i=0; i< getDictSize(); ++i) {
    D.row(i) = rae.D.row(n2o_map[i]);
  }
}

void RecursiveAutoencoderBase::averageUnknownWord() {
  D.row(0) = D.colwise().sum() / (getDictSize() - 1);
}

int RecursiveAutoencoderBase::getThetaSize() { return theta_size_; }
