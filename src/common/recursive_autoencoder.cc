// File: recursive_autoencoder.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 02-01-2013
// Last Update: Tue 13 May 2014 16:22:08 BST

#include "recursive_autoencoder.h"

#include <iostream>

// Namespaces
using namespace std;

RecursiveAutoencoderBase::RecursiveAutoencoderBase(const ModelData& config) :
  config(config), Theta_Full(0,0) { theta_ = nullptr; }


RecursiveAutoencoderBase::~RecursiveAutoencoderBase () {
  delete [] theta_;
}

void RecursiveAutoencoderBase::moveToAddress(Real* theta_address)
{
  for (int i=0; i<theta_size_; ++i) {
    theta_address[i] = theta_[i];
    // theta_[i] = 0.5;
  }
  delete [] theta_;
  theta_ = theta_address;
  // cout << "Fint " << "theta_ " << theta_ << "   : " << theta_[0] << "  " << &theta_[0] << endl;
  init(false);
}

void RecursiveAutoencoderBase::debugSize(int count)
{
  std::cout << "theta " << count << ": " << Theta_Full.sum() << endl;
}

// int RecursiveAutoencoderBase::getThetaSize() { return theta_size_; }
