// File: train_lbfgs.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 09-01-2013
// Last Update: Mon 12 May 2014 18:06:31 BST

// STL
#include <iostream>

// Boost
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>

#include "shared_defs.h"

// Local
#include "train_lbfgs.h"
#include "trainer.h"
#include "general_trainer.h"
#include "openqa_trainer.h"

#include "utils.h"
#include "recursive_autoencoder.h"
#include "finite_grad_check.h"

using namespace std;
namespace bpo = boost::program_options;


int train_lbfgs(Model& model, LineSearchType linesearch, int max_iterations, Real epsilon, Lambdas lambdas)
{
  /***************************************************************************
   *                              BFGS training                              *
   ***************************************************************************/

/*
 *   lbfgs_parameter_t param;
 *   lbfgs_parameter_init(&param);
 *   param.linesearch = linesearch;
 *   param.max_iterations = max_iterations;
 *   param.epsilon = epsilon;
 *   param.m = 25;
 *
 *   Real* vars = nullptr;
 *   int number_vars = 0;
 *
 *   model.trainer->setVarsAndNumber(vars,number_vars,model);
 *
 *   const int n = number_vars;
 *   Real error = 0.0;
 *
 *   std::random_device rd;
 *   std::mt19937 gen(rd());
 *   std::uniform_int_distribution<int> rand(1, model.to-1);
 *
 *   model.noise_sample_offset = rand(rd);
 *
 *   model.lambdas = lambdas;
 *
 *   while (model.it_count < max_iterations) {
 *     param.max_iterations = max_iterations - model.it_count;
 *     cout << "Starting new L-BFGS optimization" << endl;
 *     int ret = lbfgs(n, vars, &error, evaluate_, progress, &model, &param);
 *     cout << "L-BFGS optimization terminated with status code = " << ret << endl;
 *     cout << "fx=" << error << endl;
 *   }
 */
  return 0;
}

static int progress(
    void *instance,
    const Real *x,
    const Real *g,
    const Real fx,
    const Real xnorm,
    const Real gnorm,
    const Real step,
    int n,
    int k,
    int ls
    )
{
/*
 *   printf("Iteration %d:\n", k);
 *   printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
 *   printf("\n");
 *
 *   Model* model = reinterpret_cast<Model*>(instance);
 *   ++model->it_count;
 *   if (k % model->rae->config.dump_freq == 0)
 *   {
 *     printf("Dumping model ...\n");
 *     dumpModel(*model,k);
 *     paraphraseTest(*model,k);
 *   }
 */
  return 0;
}

Real evaluate_(
    void *instance,
    const Real *x,   // Variables (theta)
    Real *g,         // Put gradient here
    const int n,                // Number of variables
    const Real step) // line-search step used in this iteration
{
  /*
   * Model* model = reinterpret_cast<Model*>(instance);
   * BProps props(*model); // Make more efficient by creating this outside and storing it somewhere in between.
   * Real error = 0.0;
   * computeCostAndGrad(*model,x,g,n,0,props,&error);
   * return error;
   */
  return 0.0;
}


int train_lbfgs_minibatch(Model& model, LineSearchType linesearch, int
    max_iterations, Real epsilon, int batches, Lambdas lambdas)
{
  /***************************************************************************
   *                              BFGS training                              *
   ***************************************************************************/

/*
 *   lbfgs_parameter_t param;
 *   lbfgs_parameter_init(&param);
 *   param.linesearch = linesearch;
 *   param.max_iterations = 5;
 *   param.epsilon = epsilon;
 *   param.m = 5;
 *
 *   int size = int(model.corpus.size());
 *   int num_batches = min(batches,size);
 *   int batchsize = (size / num_batches) + 1;
 *
 *   cout << "Batch size: " << batchsize << " on " << num_batches << " batches." << endl;
 *   Real* vars = nullptr;
 *
 *   int number_vars = 0;
 *   model.trainer->setVarsAndNumber(vars,number_vars,model);
 *   const int n = number_vars;
 *
 *   std::random_device rd;
 *   std::mt19937 gen(rd());
 *   std::uniform_int_distribution<int> rand(1, model.to-1);
 *
 *   Real error = 0.0;
 *
 *   for (auto iteration = 0; iteration < max_iterations; ++iteration)
 *   {
 *     cout << "Iteration " << iteration << endl;
 *     std::random_shuffle ( model.indexes.begin(), model.indexes.end() );
 *     for (auto batch = 0; batch < num_batches; ++batch)
 *     {
 *       cout << "Starting next batch" << endl;
 *       model.from = batch*batchsize;
 *       model.to = min((batch+1)*batchsize,size);
 *       model.lambdas = lambdas;
 *       model.lambdas.multiply((model.to - model.from) / size);
 *       model.noise_sample_offset = rand(rd);
 *       int ret = lbfgs(n, vars, &error, evaluate_, progress_minibatch, &model, &param);
 *       cout << "L-BFGSi minibatch optimization terminated with status code = " << ret << endl;
 *       cout << "fx=" << error << endl;
 *     }
 *     if (iteration % model.rae->config.dump_freq == 0)
 *     {
 *       printf("Dumping model ...\n");
 *       dumpModel(model,iteration);
 *       paraphraseTest(model,iteration);
 *     }
 *   }
 */
  return 0;
}

static int progress_minibatch(
    void *instance,
    const Real *x,
    const Real *g,
    const Real fx,
    const Real xnorm,
    const Real gnorm,
    const Real step,
    int n,
    int k,
    int ls
    )
{
/*
 *   printf("Iteration %d:\n", k);
 *   printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
 *   printf("\n");
 *
 *   Model* model = reinterpret_cast<Model*>(instance);
 *   ++model->it_count;
 */

  return 0;
}
