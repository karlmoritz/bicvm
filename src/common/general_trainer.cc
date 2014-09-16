// File: general_trainer.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 16-01-2013
// Last Update: Mon 15 Sep 2014 14:16:27 BST

#include "general_trainer.h"

#include <iostream>
#include <random>

#include "shared_defs.h"
#include "models.h"

void GeneralTrainer::computeCostAndGrad( Model& model, const Real* x, Real* gradient_location,
                        int n, int iteration, BProps& props, Real* error)
{
  props.propA->reset();
  if (props.propB != nullptr) { props.propB->reset(); }
  if (props.docprop != nullptr) { props.docprop->propA->reset(); props.docprop->propB->reset(); }

  /******************************************************************************
   *                Check for bimodel setup - otherwise proceed                 *
   ******************************************************************************/
  if (model.b != nullptr) {
    computeBiCostAndGrad(model, *model.b, x, gradient_location, n, iteration, props, error);
    return;
  }

  WeightVectorType weights(gradient_location,n);
#pragma omp single
  {
    weights.setZero();
  }

  BackpropagatorBase* lblprop = (model.rae->config.calc_lbl) ? model.rae->getBackpropagator(model,n) : nullptr;
  BackpropagatorBase* raeprop = (model.rae->config.calc_rae) ? model.rae->getBackpropagator(model,n) : nullptr;
  BackpropagatorBase* uaeprop = (model.rae->config.calc_uae) ? model.rae->getBackpropagator(model,n) : nullptr;

  size_t thread_num = omp_get_thread_num();
  size_t num_threads = omp_get_num_threads();
  for (size_t i = model.from + thread_num; i < model.to; i += num_threads) {
    int j = model.indexes[i];
    VectorReal x(model.rae->config.word_representation_size);
    if (model.rae->config.calc_lbl) lblprop->backPropagateLbl(j,&x);
    if (model.rae->config.calc_rae) raeprop->backPropagateRae(j,&x);
    if (model.rae->config.calc_uae) uaeprop->backPropagateUnf(j,&x);

#pragma omp critical
    {
      // BROKEN: Separate weights from dict weights!
      if (model.rae->config.calc_lbl) {
        weights += lblprop->dumpWeights();
        *error += lblprop->getError();
      }
      if (model.rae->config.calc_rae) {
        weights += raeprop->dumpWeights();
        *error += raeprop->getError();
      }
      if (model.rae->config.calc_uae) {
        weights += uaeprop->dumpWeights();
        *error += uaeprop->getError();
      }
    }

#pragma omp master
    {
#pragma omp critical
      {
      if (model.calc_L2)  *error += model.rae->getLambdaCost(model.bools, model.lambdas);
      if (model.calc_L2)  model.rae->addLambdaGrad(gradient_location,model.bools, model.lambdas);
      }

      if (model.rae->config.calc_lbl) lblprop->printInfo();
      int outwidth = 16;
      // cout << setw(outwidth) << "ERRORS";
      // if (model.rae->config.calc_lbl) cout << setw(outwidth) << "A(lbl)";
      // if (model.rae->config.calc_rae) cout << setw(outwidth) << "A(rae)";
      // cout << endl;
      // cout << setw(outwidth) << " ";
      // if (model.rae->config.calc_lbl) cout << setw(outwidth) << lblprop->getError();
      // if (model.rae->config.calc_rae) cout << setw(outwidth) << raeprop->getError();
      // cout << endl;
    }
    if (model.rae->config.calc_lbl) delete lblprop;
    if (model.rae->config.calc_rae) delete raeprop;
    if (model.rae->config.calc_uae) delete uaeprop;
  }
}

void GeneralTrainer::computeBiCostAndGrad(Model &modelA, Model &modelB, const Real *x,
                          Real *gradient_location, int n, int iteration,
                          BProps &prop, Real* error) {

  int modsize_A = modelA.rae->getThetaSize();
  int modsize_B = modelB.rae->getThetaSize();
  int dictsize_A = modelA.rae->de_->getThetaSize();
  int dictsize_B = modelB.rae->de_->getThetaSize();

  // Create gradient vectors for modA, modB, dictA, dictB
  Real* ptr = gradient_location;
  WeightVectorType weightsA(ptr,modsize_A); ptr += modsize_A;
  WeightVectorType weightsB(ptr,modsize_B); ptr += modsize_B;
  WeightVectorType dweightsA(ptr,dictsize_A); ptr += dictsize_A;
  WeightVectorType dweightsB(ptr,dictsize_B); ptr += dictsize_B;
  assert (gradient_location + n == ptr);

#pragma omp single
  {
    weightsA.setZero();
    weightsB.setZero();
    dweightsA.setZero();
    dweightsB.setZero();
  }

  WeightMatrixType docgrad_AD(0,0,0);
  WeightMatrixType docgrad_BD(0,0,0);
  if (modelA.docmod != nullptr) {
    int modsize_C = modelA.docmod->rae->getThetaSize();
    int modsize_D = modelB.docmod->rae->getThetaSize();
    int dictsize_C = modelA.docmod->rae->de_->getThetaSize();
    int dictsize_D = modelB.docmod->rae->de_->getThetaSize();

    int word_width = modelA.rae->config.word_representation_size;
    int docAdict_size = modelA.docmod->rae->getDictSize();
    int docBdict_size = modelB.docmod->rae->getDictSize();
    // Bonus weights even further back. Use these for pulling docmodgrads.
    // [modA,modB,dicA,dicB,modC,modD,dicC,dicB]
    ptr += modsize_C + modsize_D;
    new (&docgrad_AD) WeightMatrixType(ptr, docAdict_size, word_width);
    ptr += dictsize_C;
    new (&docgrad_BD) WeightMatrixType(ptr, docBdict_size, word_width);
    ptr += dictsize_D;
  }
  assert (gradient_location + n == ptr);

  Real gamma = modelA.gamma;

  // Iterating over all sentences in the main corpus.
  size_t thread_num = omp_get_thread_num();
  size_t num_threads = omp_get_num_threads();
  for (size_t i = modelA.from + thread_num; i < modelA.to; i += num_threads) {
    int j = modelA.indexes[i];

    VectorReal rootA(modelA.rae->config.word_representation_size);
    VectorReal rootB(modelA.rae->config.word_representation_size);

    if (modelA.rae->config.calc_lbl) prop.propA->backPropagateLbl(j,&rootA);
    if (modelA.rae->config.calc_rae) prop.propA->backPropagateRae(j,&rootA);
    if (modelB.rae->config.calc_rae) prop.propB->backPropagateRae(j,&rootB);
    if (modelA.rae->config.calc_uae) prop.propA->backPropagateUnf(j,&rootA);
    if (modelB.rae->config.calc_uae) prop.propB->backPropagateUnf(j,&rootB);

    if (modelA.rae->config.calc_bi) {
      // Forward propagate parallel and noise version (B)
      SinglePropBase* other = prop.propB->forwardPropagate(j,&rootB);

      if (modelB.docmod != nullptr) {
        // If docmod, pass the sentence vector into the docmod model now.
        // sent_id is unique, so parallel access should not be an issue.
        int sent_id = modelB.corpus.id[j];
        modelB.docmod->rae->de_->D.row(sent_id) = rootB;
      }

      // The "normal" biprop: backprop self given the other root and vice versa
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
        SinglePropBase* noise = prop.propB->forwardPropagate(noise_int, &noise_root);

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
      if (noise_count > 0 || (modelA.docmod != nullptr && iteration > 0)) {
        prop.propA->addError(0.5*gamma*noise_error);
        prop.propB->addError(0.5*gamma*noise_error);
        SinglePropBase* selff = prop.propA->forwardPropagate(j,&rootA);
        // dH/dA = N - B
        if (modelA.docmod != nullptr && iteration > 0) {
          // Add docmod gradient and error!
          prop.propA->backPropagateGiven(j, selff,
                                         docgrad_AD.row(modelA.corpus.id[j]).transpose() +
                                         gamma*(combined_noise_root - noise_count*rootB));
        } else {
          prop.propA->backPropagateGiven(j, selff,
                                         gamma*(combined_noise_root - noise_count*rootB));
        }
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
      if (modelA.docmod != nullptr) {
        // If docmod, pass the sentence vector into the docmod model now.
        int sent_id = modelA.corpus.id[j];
        modelA.docmod->rae->de_->D.row(sent_id) = rootA; // using sent_id directly to store correct row.
      }

      // Do the "normal" biprop: backprop self given the other root and vice
      prop.propB->backPropagateBi(j,&rootB,rootA); // inefficient (repeats fprop)
      prop.propA->backPropagateBi(j,&rootA,rootB);

      VectorReal combined_noise_root(modelA.rae->config.word_representation_size);
      combined_noise_root.setZero();
      Real noise_error = 0.0;
      int noise_count = 0.0;

      for (int n = 0; n < modelA.num_noise_samples; ++n) {
        int noise_int =  modelA.indexes[
          (
              i - modelA.from + 1 +
              ( modelA.noise_sample_offset*(n+1) )%(modelA.to-modelA.from-1)
          ) % (modelA.to-modelA.from)
          +modelA.from];

        VectorReal noise_root(modelA.rae->config.word_representation_size);
        SinglePropBase* noise = prop.propA->forwardPropagate(noise_int, &noise_root);
        Real hinge = modelA.rae->config.hinge_loss_margin + 0.5 * (rootB - rootA).squaredNorm() - 0.5 * (rootB - noise_root).squaredNorm();
        if (hinge > 0) {
          noise_error += hinge;
          ++noise_count;
          combined_noise_root += noise_root;
          prop.propA->backPropagateGiven(noise_int,noise,gamma*(rootB - noise_root));
          prop.propA->addCountsAndGradsForGiven(noise_int, noise);
        }
      }
      if (noise_count > 0 || (modelB.docmod != nullptr && iteration > 0)) {
        prop.propB->addError(0.5*gamma*noise_error);
        prop.propA->addError(0.5*gamma*noise_error);
        // Add docmod gradient and error!
        SinglePropBase* selff = prop.propB->forwardPropagate(j,&rootB);
        if (modelB.docmod != nullptr && iteration > 0) {
          // Add docmod gradient and error!
          prop.propB->backPropagateGiven(j, selff,
                                         docgrad_BD.row(modelB.corpus.id[j]).transpose() +
                                         gamma*(combined_noise_root - noise_count*rootA));
        } else {
          prop.propB->backPropagateGiven(j,selff,
                                         gamma*(combined_noise_root - noise_count*rootA));
        }
        prop.propB->addCountsAndGradsForGiven(j, selff);
        other = prop.propA->forwardPropagate(j,&rootA);
        prop.propA->backPropagateGiven(j,other,gamma*noise_count*(rootA - rootB));
        prop.propA->addCountsAndGradsForGiven(j, other);
      }
    }

    if (modelA.rae->config.calc_through) {
      // TODO(kmh): This is almost certainly buggy!
      SinglePropBase* other = prop.propA->forwardPropagate(j, &rootB); // forward propagates (sets &rootB to root)
      prop.propB->unfoldPropagateGiven(j, other, &rootA); // unfold using other->root as root, sets rootA to delta_D[0]
      prop.propA->backPropagateGiven(j, other, rootA); // backprops given model (other) with gradient (rootA)
      prop.propA->addCountsAndGradsForGiven(j, other);
    }
    if (modelB.rae->config.calc_through) {
      SinglePropBase* other = prop.propB->forwardPropagate(j, &rootB); // forward propagates (on B)
      prop.propA->unfoldPropagateGiven(j, other, &rootA); // unfold given a root node, calculates sets gradient (A given B)
      prop.propB->backPropagateGiven(j, other, rootA); // backprop given a gradient on the root node (B given grad(A) )
      prop.propB->addCountsAndGradsForGiven(j, other);
    }
  }  // end of for loop over sentences.

#pragma omp master
  {
    // if (modelA.rae->config.calc_lbl) prop.propA->printInfo();
    int outwidth = 16;
    // cout << setw(outwidth) << "ERRORS";
    // cout << setw(outwidth) << "A";
    // cout << setw(outwidth) << "B";
    // cout << endl;

    // cout << setw(outwidth) << " ";
    // cout << setw(outwidth) << prop.propA->getError();
    // cout << setw(outwidth) << prop.propB->getError();
    // cout << endl;
  }

#pragma omp critical
  {
    *error += prop.propA->getError();
    *error += prop.propB->getError();

    weightsA += prop.propA->dumpWeights();
    weightsB += prop.propB->dumpWeights();
    dweightsA += prop.propA->dumpDict();
    dweightsB += prop.propB->dumpDict();
  }

#pragma omp single
  {
    // L2 cost after normalization.
#pragma omp critical
    {
      // speedup if I make this "single nowait" and dump onto my own bprop?
      // would require pushing Real* data into bpropbase.
    if (modelA.calc_L2) *error += modelA.rae->getLambdaCost(modelA.bools, modelA.lambdas);
    if (modelB.calc_L2) *error += modelB.rae->getLambdaCost(modelB.bools, modelA.lambdas);
    if (modelA.calc_L2) *error += modelA.rae->de_->getLambdaCost(modelA.bools, modelA.lambdas);
    if (modelB.calc_L2) *error += modelB.rae->de_->getLambdaCost(modelB.bools, modelA.lambdas);

    ptr = gradient_location;
    if (modelA.calc_L2) modelA.rae->addLambdaGrad(ptr, modelA.bools, modelA.lambdas);
    ptr += modsize_A;
    if (modelB.calc_L2) modelB.rae->addLambdaGrad(ptr, modelB.bools, modelA.lambdas);
    ptr += modsize_B;
    if (modelA.calc_L2) modelA.rae->de_->addLambdaGrad(ptr, modelA.bools, modelA.lambdas);
    ptr += dictsize_A;
    if (modelB.calc_L2) modelB.rae->de_->addLambdaGrad(ptr, modelB.bools, modelA.lambdas);
    ptr += dictsize_B;
    }
  }

  // If we're at the final loop (in case of minibatch updates) we now calculate
  // the gradients for the document level model.
  if (modelA.to == modelA.corpus.size() && modelA.docmod != nullptr) {
    // MODIFY THIS TO SET modelA.docmod.to and .from
    // TODO TODO TODO
    int docmodsize = modelA.docmod->rae->getThetaSize()
      + modelB.docmod->rae->getThetaSize()
      + modelA.docmod->rae->de_->getThetaSize()
      + modelB.docmod->rae->de_->getThetaSize();
    computeBiCostAndGrad(*modelA.docmod, *modelB.docmod, x, ptr, docmodsize, 1,
                         *prop.docprop, error);
  }
}

void GeneralTrainer::testModel(Model &model) {

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
    SinglePropBase* propagator = model.rae->getSingleProp(model.corpus.words[j].size(),
        model.corpus.nodes[j].size(),
        0.5,model.bools);
    propagator->loadWithSentence(model.corpus, j);
    propagator->forwardPropagate(false);
#pragma omp critical
    {
      correctly_classified_sent += propagator->evaluateSentence();
    }
    delete propagator;
  }
// #pragma omp master
  // {
    // cout << "  " << correctly_classified_sent << "/" << num_sentences << " ";
  // }
}

void GeneralTrainer::setVarsAndNumber(Real *&vars, int &number_vars, Model &model) {
  number_vars += model.rae->getThetaSize();
  number_vars += model.rae->de_->getThetaSize();
  vars = model.rae->theta_;

  if (model.b != nullptr)
  {
    number_vars += model.b->rae->getThetaSize();
    number_vars += model.b->rae->de_->getThetaSize();
  }

}
