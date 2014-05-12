// File: train_adagrad.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 01-01-2013
// Last Update: Mon 12 May 2014 18:00:02 BST

// STL
#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>

// Boost
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>

// Local
#include "train_adagrad.h"
#include "trainer.h"
#include "general_trainer.h"
#include "openqa_trainer.h"
#include "recursive_autoencoder.h"
#include "utils.h"
#include "fast_math.h"

using namespace std;
namespace bpo = boost::program_options;


int train_adagrad(Model &model, int iterations, Real eta, Model *tmodel, int batches, Lambdas lambdas, Real l1)
{
  Real* vars = nullptr;
  int number_vars = 0;
  model.trainer->setVarsAndNumber(vars,number_vars,model);
  WeightArrayType theta(vars,number_vars);
  Real* Gt_d = new Real[number_vars]();
  WeightArrayType Gt(Gt_d,number_vars);
  int extended_vars = number_vars;
  if (model.docmod != nullptr) {
    extended_vars += model.docmod->rae->getThetaSize() +
      model.b->docmod->rae->getThetaSize();
  }
  Real* gradient = new Real[extended_vars]();

  /*
   * // Remove L2 regularization as AdaGrad uses L1 instead
   * model.calc_L2 = false;
   * if (model.b != nullptr)
   *   model.b->calc_L2 = false;
   */

  int size = int(model.corpus.size());
  int num_batches = min(batches,size/2);
  int batchsize = max((size/num_batches),2);
  cout << "Batch size: " << batchsize << "  eta " << eta << endl;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> rand(1, batchsize-1);

  Real error;

#pragma omp parallel
  {
    Real update;
    Real l1_reg, l1_batch;
    BProps props = BProps(model);
    for (auto iteration = 0; iteration < iterations; ++iteration) {
#pragma omp master
      {
        cout << "Iteration " << iteration << endl;
      }
      // std::random_shuffle ( model.indexes.begin(), model.indexes.end() );
      for (auto batch = 0; batch < num_batches; ++batch) {
#pragma omp single
        {
          model.noise_sample_offset = rand(rd);
          model.from = batch*batchsize;
          model.to = min((batch+1)*batchsize,size);
          model.lambdas = lambdas;
          model.lambdas.multiply((model.to - model.from) / size);
          error = 0.0;
        }
        l1_batch = l1 * (model.to - model.from) / size;
        model.trainer->computeCostAndGrad(model,nullptr,gradient,number_vars,iteration,props,&error);
#pragma omp barrier
// #pragma omp single
        // {
          // seeing that I need to iterate ...
          // Update from Duchi et al. 2010: 24
          // x_{t+1,i} = sign(-g_{t,i}) \frac_{\eta t}_{H_{t,ii}} \[ \|g_{t,i}\| - \lambda \]_+
          // Update  = sign(-g_{t,i}) \frac_{\eta t}_{H_{t,ii}} \|g_{t,i}\|
          // Regular = sign(-g_{t,i}) \frac_{\eta t}_{H_{t,ii}} \lambda
          size_t thread_num = omp_get_thread_num();
          size_t num_threads = omp_get_num_threads();
          for (size_t i = 0 + thread_num; i < number_vars; i += num_threads) {
          // for (int i = 0; i < number_vars; ++i) {
            Gt_d[i] += gradient[i]*gradient[i];
            // Update weight: ( eta / \sqrt(Sum square gradients) ) * gradient
            if (abs(Gt_d[i]) > 0.0000000000000001) {
              update = vars[i] - ((eta / (sqrt(Gt_d[i]))) * gradient[i]);
              if (l1_batch > 0.0) {
                l1_reg = (eta / (sqrt(Gt_d[i]))) * l1_batch;
                vars[i] = signum(update) * max(Real(0.0), abs(update) - l1_reg);
              } else {
                vars[i] = update;
              }
            }
          }
        // }
#pragma omp barrier
      }
      if (tmodel != nullptr) {
#pragma omp single
        {
          cout << "Correct";
          model.trainer->testModel(*tmodel);
          cout << endl;
        }
      }
      if ((iteration+1) % model.rae->config.dump_freq == 0) {
#pragma omp single
        {
          printf("Dumping model ...\n");
          dumpModel(model,iteration);
          paraphraseTest(model,iteration);
        }
      }
    }
  }
  delete [] gradient;
  delete [] Gt_d;
  return 0;
}

