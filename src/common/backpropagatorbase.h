// File: backpropagatorbase.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 22-04-2013
// Last Update: Thu 29 May 2014 15:03:16 BST

#ifndef COMMON_BACKPROPAGATORBASE_H
#define COMMON_BACKPROPAGATORBASE_H

// Local
#include "shared_defs.h"
#include "recursive_autoencoder.h"
#include "singlepropbase.h"

class Trainer;
class GeneralTrainer;
class OpenQATrainer;

class BackpropagatorBase
{
public:
  BackpropagatorBase (RecursiveAutoencoderBase* rae, const Model &model);
  virtual ~BackpropagatorBase ();

  // Pure virtual functions.

  // Normalizes gradients in internal storage. Type parameter informs
  // normalization function of the type of propagation the propagator was used
  // for (type: 0=rae, 1=lbl, 2=bi).
  virtual void normalize(int type) = 0; // type: 0=rae, 1=lbl, 2=bi
  virtual void collectGradients(SinglePropBase* spb, int sentence) = 0;

  // Common functionality.
  Real getError();
  void addError(Real i);
  void reset() { error_ = 0; weights.setZero(); dict_weights.setZero(); } // subsequently add counts from below, too
  WeightVectorType dumpWeights();
  WeightVectorType dumpDict();
  void printInfo();

  // Propagation Functions.

  // Forward propagates, returns propagated singleprop for further manipulation.
  SinglePropBase* forwardPropagate(int i, VectorReal* x);

  // Forward propagates a sentence that is 66% noise and 33% truth or such.
  SinglePropBase* noisyForwardPropagate(int noise, int truth, VectorReal* x);

  // Forward propagates, backprop with autoencoder error. Stores error and
  // gradients in internal variables and sets x to the root encoding.
  void backPropagateRae(int i, VectorReal* x);
  void backPropagateUnf(int i, VectorReal* x);
  int backPropagateLbl(int i, VectorReal* x);
  // Word could be const, but can't figure out how to do it.
  void backPropagateBi(int i,  VectorReal* x, VectorReal word);

  // backprop given a gradient on the root node (B given grad(A) )
  void backPropagateGiven(int i, SinglePropBase* other, const VectorReal& gradient);
  void addCountsAndGradsForGiven(int i, SinglePropBase* thisModel);

  // unfold given a root node, calculates sets gradient (A given B)
  void unfoldPropagateGiven(int i, SinglePropBase* other, VectorReal* gradient);

  friend class Trainer;
  friend class OpenQATrainer;
  friend class GeneralTrainer;
  /* friend void computeBiCostAndGrad(Model& modelA, Model& modelB, const Real* x, */
                                   /* Real* gradient_location, int n, */
                                   /* int iteration, BProps& prop, Real* error); */
  /* friend void computeCostAndGrad( Model &model, */
                                            /* const Real* x, */
                                            /* Real* gradient_location, */
                                            /* int n, int iteration, BProps& prop, */
                                            /* Real* error); */

protected:
  RecursiveAutoencoderBase* rae_;
  const Model& model;
  WeightVectorType weights;
  WeightVectorType dict_weights;
  Real error_;
  int word_width;
  int dict_size;

  Sentence corrupt;

  // Counter for nodes and words used for normalization.
  int count_nodes_;
  int count_words_;

  // Some counts for printing statistics.
  int correctly_classified_sent;
  int zero_should_be_one;
  int zero_should_be_zero;
  int one_should_be_zero;
  int one_should_be_one;
  int is_a_zero;
  int is_a_one;

  SinglePropBase* singleprop;

private:
  void backPropagateAEWrapper(int i, bool unfold, VectorReal* x);
};

#endif  // COMMON_BACKPROPAGATORBASE_H
