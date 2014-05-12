// File: train_sgd.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 01-01-2013
// Last Update: Thu 02 Jan 2014 03:35:45 PM GMT
/*------------------------------------------------------------------------
 * Description: <DESC>
 *
 *------------------------------------------------------------------------
 * History:
 * TODO:
 *========================================================================
 */

// STL
#include <iostream>
#include <cmath>

// Boost
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>

#include "shared_defs.h"

// L-BFGS
#include <lbfgs.h>

// Local
#include "train_sgd.h"
#include "train_update.h"
#include "recursive_autoencoder.h"
#include "utils.h"

using namespace std;
namespace bpo = boost::program_options;


int train_sgd(Model &model, int iterations, Real eta, Lambdas lambdas)
{
  Real* vars = nullptr;
  int number_vars = 0;

  setVarsAndNumber(vars,number_vars,model);
  cout << vars;

  // Real eta_t0 = eta;
  WeightVectorType theta(vars,number_vars);

  Real* dataOne = new Real[number_vars];
  Real* dataTwo = new Real[number_vars];
  Real* data1 = dataOne;
  // Real* data2 = dataTwo;
  Real error = 100000;
  Real new_error = 100000;

  model.lambdas = lambdas;
  BProps props = BProps(model);

  for (auto iteration = 0; iteration < iterations; ++iteration)
  {
    model.rae->debugSize(3);
    WeightVectorType grad(data1,number_vars);
    Real new_error = 0.0;
    computeCostAndGrad(model,nullptr,data1,number_vars,iteration,props,&new_error);
    cout << "Correct (error): " << new_error << " ... etc: " << eta << endl;
    cout << "Grad sum " << grad.sum();
    cout << "Theta sum " << theta.sum();

    grad *= eta;
    theta -= grad;

    if (data1 == dataOne) { data1 = dataTwo; } // data2 = dataOne; }
    else { data1 = dataOne; } //data2 = dataTwo; }

    if (new_error > error)
      eta *= 0.5;
    else
      eta *= 1.05;
    error = new_error;
    // if (eta*1000 < eta_t0)
    // eta = eta_t0; // Hack: if eta gets too small, we reset it and jump out of misery
    paraphraseTest(model,iteration);
  }

  delete [] dataOne;
  delete [] dataTwo;
  return 0;
}

