// File: openqa_fast_bordes_trainer.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 16-01-2013
// Last Update: Thu 05 Jun 2014 09:23:12 BST

#include "openqa_fast_bordes_trainer.h"

#include <iostream>
#include <random>

#include "shared_defs.h"
#include "models.h"

void OpenQAFastBordesTrainer::computeCostAndGrad( Model& model, const Real* x, Real* gradient_location,
                        int n, int iteration, BProps& props, Real* error)
{
  assert(props.propB != nullptr);
  assert(props.docprop != nullptr);

  props.propA->reset();
  props.propB->reset();
  props.docprop->propA->reset();
  props.docprop->propB->reset();

  WeightVectorType zeroMe(gradient_location,n); zeroMe.setZero(); // set gradients to zero.

  /*
   * if (iteration % 2 == 0) {
   *   // Question - Query
   * }
   * else {
   */
    // Question - Question Paraphrases
    int modsize_A = model.rae->getThetaSize();
    int modsize_B = model.b->rae->getThetaSize();
    Real* ptr = gradient_location;
    ptr += modsize_A + modsize_B;
    int m = n - modsize_A - modsize_B;
    computeBiCostAndGrad(*model.docmod, *(model.b->docmod), ptr, m, iteration, *props.docprop, error, false);
  /*
   * }
   */
    computeBiCostAndGrad(model, *model.b, gradient_location, n, iteration, props, error, true);
}

void OpenQAFastBordesTrainer::computeBiCostAndGrad(Model &modelA, Model &modelB,
                          Real *gradient_location, int n, int iteration,
                          BProps &prop, Real* error, bool near_noise) {

  int modsize_A = modelA.rae->getThetaSize();
  int modsize_B = modelB.rae->getThetaSize();
  int dictsize_A = modelA.rae->de_->getThetaSize();

  // Create gradient vectors for modA, modB, dictA
  Real* ptr = gradient_location;
  WeightVectorType weightsA(ptr,modsize_A); ptr += modsize_A;
  WeightVectorType weightsB(ptr,modsize_B); ptr += modsize_B;
  if (modelA.docmod != nullptr) {
    // If we have docmods, we need to skip over those for the placement.
    // This will be the case if we train on question-query, but not if we train
    // on the paraphrase data.
    ptr += modelA.docmod->rae->getThetaSize();
    ptr += modelB.docmod->rae->getThetaSize();
  }
  WeightVectorType dweightsA(ptr,dictsize_A); ptr += dictsize_A;
  assert (gradient_location + n == ptr);

/*
 * #pragma omp single
 *   {
 *     weightsA.setZero();
 *     weightsB.setZero();
 *     dweightsA.setZero();
 *   }
 */

  Real gamma = modelA.gamma;

  // Iterating over all sentences in the main corpus.
  size_t thread_num = omp_get_thread_num();
  size_t num_threads = omp_get_num_threads();
  for (size_t i = modelA.from + thread_num; i < modelA.to; i += num_threads) {
    int j = modelA.indexes[i];

    VectorReal rootA(modelA.rae->config.word_representation_size);
    VectorReal rootB(modelA.rae->config.word_representation_size);

    if (modelA.rae->config.calc_bi) {
      // Forward propagate parallel and noise version (B)
      SinglePropBase* other = prop.propB->forwardPropagate(j,&rootB);

      // The "normal" biprop: backprop self given the other root and vice versa
      // ARGH ARGH ARGH ARGH
      prop.propA->backPropagateBi(j,&rootA,rootB); // inefficient (repeats fprop)
      prop.propB->backPropagateBi(j,&rootB,rootA);

      VectorReal combined_noise_root(modelA.rae->config.word_representation_size);
      combined_noise_root.setZero();
      Real noise_error = 0.0;
      int noise_count = 0;

      for (int n = 0; n < modelA.num_noise_samples; ++n) {
        int noise_int = modelA.indexes[
          (
              i - modelA.from + 1 +
              ( modelA.noise_sample_offset*(n+1) )%(modelA.to-modelA.from-1)
          ) % (modelA.to-modelA.from)
          +modelA.from];

        VectorReal noise_root(modelA.rae->config.word_representation_size);
        SinglePropBase* noise = (near_noise)
          ? prop.propB->noisyForwardPropagate(noise_int, j, &noise_root)
          : prop.propB->forwardPropagate(noise_int, &noise_root);

        Real hinge = modelA.rae->config.hinge_loss_margin + 0.5 * (rootA - rootB).squaredNorm() - 0.5 * (rootA - noise_root).squaredNorm();
        if (hinge > 0) {
          noise_error += hinge;
          ++noise_count;
          combined_noise_root += noise_root;
          // dH/dN = A - N
          prop.propB->backPropagateGiven(noise_int,noise,gamma*(rootA - noise_root));
          prop.propB->addCountsAndGradsForGiven(noise_int, noise);
        }
      }
      if (noise_count > 0) {
        prop.propA->addError(0.5*gamma*noise_error);
        prop.propB->addError(0.5*gamma*noise_error);
        SinglePropBase* selff = prop.propA->forwardPropagate(j,&rootA);
        // dH/dA = N - B
          prop.propA->backPropagateGiven(j, selff,
                                         gamma*(combined_noise_root - noise_count*rootB));
        prop.propA->addCountsAndGradsForGiven(j, selff);
        // dH/dB = B - A
        other = prop.propB->forwardPropagate(j,&rootB);
        prop.propB->backPropagateGiven(j,other,gamma*noise_count*(rootB - rootA));
        prop.propB->addCountsAndGradsForGiven(j, other);
      }
    }

    if (modelB.rae->config.calc_bi) {
      // Forward propagate parallel and noise version (B)
      SinglePropBase* other = prop.propA->forwardPropagate(j,&rootA);

      // Do the "normal" biprop: backprop self given the other root and vice
      prop.propB->backPropagateBi(j,&rootB,rootA); // inefficient (repeats fprop)
      prop.propA->backPropagateBi(j,&rootA,rootB);

      VectorReal combined_noise_root(modelA.rae->config.word_representation_size);
      combined_noise_root.setZero();
      Real noise_error = 0.0;
      int noise_count = 0.0;

      for (int n = 0; n < modelA.num_noise_samples; ++n) {
        int noise_int = modelA.indexes[
          (
              i - modelA.from + 1 +
              ( modelA.noise_sample_offset*(n+1) )%(modelA.to-modelA.from-1)
          ) % (modelA.to-modelA.from)
          +modelA.from];

        VectorReal noise_root(modelA.rae->config.word_representation_size);
        SinglePropBase* noise = prop.propA->forwardPropagate(noise_int, &noise_root);
        /*
         * SinglePropBase* noise = (near_noise)
         *   ? prop.propA->noisyForwardPropagate(noise_int, j, &noise_root)
         *   : prop.propA->forwardPropagate(noise_int, &noise_root);
         */

        Real hinge = modelA.rae->config.hinge_loss_margin + 0.5 * (rootB - rootA).squaredNorm() - 0.5 * (rootB - noise_root).squaredNorm();
        if (hinge > 0) {
          noise_error += hinge;
          ++noise_count;
          combined_noise_root += noise_root;
          prop.propA->backPropagateGiven(noise_int,noise,gamma*(rootB - noise_root));
          prop.propA->addCountsAndGradsForGiven(noise_int, noise);
        }
      }
      if (noise_count > 0) {
        prop.propB->addError(0.5*gamma*noise_error);
        prop.propA->addError(0.5*gamma*noise_error);
        // Add docmod gradient and error!
        SinglePropBase* selff = prop.propB->forwardPropagate(j,&rootB);
          prop.propB->backPropagateGiven(j,selff,
                                         gamma*(combined_noise_root - noise_count*rootA));
        prop.propB->addCountsAndGradsForGiven(j, selff);
        other = prop.propA->forwardPropagate(j,&rootA);
        prop.propA->backPropagateGiven(j,other,gamma*noise_count*(rootA - rootB));
        prop.propA->addCountsAndGradsForGiven(j, other);
      }
    }
  }  // end of for loop over sentences.

#pragma omp critical
  {
    *error += prop.propA->getError();
    *error += prop.propB->getError();

    weightsA += prop.propA->dumpWeights();
    weightsB += prop.propB->dumpWeights();
    dweightsA += prop.propA->dumpDict();
    dweightsA += prop.propB->dumpDict();
  }

#pragma omp single
  {
    // L2 cost after normalization.
// #pragma omp critical // WHY THE CRITICAL ON TOP?
    // {
      if (modelA.calc_L2) *error += modelA.rae->getLambdaCost(modelA.bools, modelA.lambdas);
      if (modelB.calc_L2) *error += modelB.rae->getLambdaCost(modelB.bools, modelA.lambdas);

      ptr = gradient_location;
      if (modelA.calc_L2) modelA.rae->addLambdaGrad(ptr, modelA.bools, modelA.lambdas);
      ptr += modsize_A;
      if (modelB.calc_L2) modelB.rae->addLambdaGrad(ptr, modelB.bools, modelA.lambdas);
      ptr += modsize_B;

      if (modelA.docmod) {
        // If we have docmods we
        //  (a) need to skip over those to get to the dictionary.
        //  (b) consider the dictionary L2 (to make sure we only do this once)
        ptr += modelA.docmod->rae->getThetaSize();
        ptr += modelB.docmod->rae->getThetaSize();
        if (modelA.calc_L2) *error += modelA.rae->de_->getLambdaCost(modelA.bools, modelA.lambdas);
        if (modelA.calc_L2) modelA.rae->de_->addLambdaGrad(ptr, modelA.bools, modelA.lambdas);
      }
    // }
  }
}

void OpenQAFastBordesTrainer::testModel(Model &model) {

  /***************************************************************************
   *             Define a couple of frequently needed variables              *
   ***************************************************************************/

  int num_sentences = model.rae->config.num_sentences;
  if (num_sentences == 0)
    num_sentences = int(model.corpus.size());
  else
    num_sentences = min(num_sentences,int(model.corpus.size()));

  int correctly_classified_sent = 0;

#pragma omp parallel for schedule(dynamic)
  for (auto i = 0; i<num_sentences; ++i) {
    int j = model.indexes[i];
    SinglePropBase* propagator = model.rae->getSingleProp(model.corpus[j].words.size(),
                                                          model.corpus[j].nodes.size(),
                                                          0.5,model.bools);
    propagator->loadWithSentence(model.corpus[j]);
    propagator->forwardPropagate(false);
#pragma omp critical
    {
      correctly_classified_sent += propagator->evaluateSentence();
    }
    delete propagator;
  }
}

void OpenQAFastBordesTrainer::setVarsAndNumber(Real *&vars, int &number_vars, Model &model) {
  number_vars += model.rae->getThetaSize();
  number_vars += model.b->rae->getThetaSize();
  number_vars += model.docmod->rae->getThetaSize();
  number_vars += model.b->docmod->rae->getThetaSize();
  number_vars += model.rae->de_->getThetaSize();
  vars = model.rae->theta_;
}
