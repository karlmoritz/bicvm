// File: singlepropbase.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 03-01-2013
// Last Update: Mon 15 Sep 2014 14:33:51 BST

#ifndef COMMON_SINGLEPROPBASE_H
#define COMMON_SINGLEPROPBASE_H

#include "shared_defs.h"

#include <exception>
// #include "recursive_autoencoder.h"

// class RecursiveAutoencoderBase;
class BackpropagatorBase;

class SinglePropBase
{
 public:
  SinglePropBase(Bools updates, Real beta,
                 bool param_on_tree, bool has_unfolding);
  virtual ~SinglePropBase();
  virtual void loadWithSentence(const Corpus& c, int i);


  // Common propagation functions.
  virtual void forwardPropagate(bool autoencode = false);
  virtual int backPropagate(bool lbl_error,
                            bool rae_error,
                            bool bi_error,
                            bool unf_error,
                            VectorReal *word = nullptr);
  void unfoldFromHere(int node);
  void backpropAllWords();

  // Common functions.
  int evaluateSentence();
  void setToD(VectorReal* x, int i);
  void setDynamic(WeightVectorType& dynamic, int mode = 0);

  Real getLblError();
  Real getRaeError();

  int getJointNodes();
  int getClassCorrect();

  int getSentLength();
  int getNodesLength();

  virtual void passDictLink(Real* data, int size) {};
  virtual void passDataLink(Real* data, int size) {};

 protected:
  // Individual propagation steps.
  virtual void encodeInputs(int i, int child0, int child1, int rule, int rc0,
                            int rc1, bool autoencode) = 0;
  virtual void encodeSingular(int i, int child0, int rule, int rc0,
                              bool autoencode) = 0;
  virtual int applyLabel(int parent, bool updateWcat, Real beta) = 0;
  virtual void backpropInputs(int node, int child0, int child1, int rule,
                              int rc0, int rc1, bool rae_error,
                              bool unf_error) = 0;
  virtual void backpropWord(int node, int sent_pos) = 0;
  virtual void backpropBi(int node, VectorReal* word) = 0;

  // Additional propagation steps for unfolding autoencoders.
  virtual void unfoldProp(int node, int child0, int child1, int rule, int rc0,
                          int rc1) { throw ENotImplemented(); };
  virtual void unfoldRecError(int node) { throw ENotImplemented(); };
  virtual void unfoldBackprop(int node, int child0, int child1, int rule,
                              int rc0, int rc1) { throw ENotImplemented(); };

  // Simple variables.
  int sent_length;
  int nodes_length;
  int word_width;
  // const Sentence * instance_; // instead, link directly to more complex forms
  // const vector<LabelID> * words_;
  const Corpus * corpus_; // CORPUS link
  int id;                 // sentence in corpus
  // int value_;
  // when needed!
  Bools updates;
  Real beta_;

  // Errors
  Real lbl_error_;  // label error
  Real rae_error_;  // autoencoder error

  // Count of correctly and wrongly classified nodes in tree.
  int classified_correctly_;
  int classified_wrongly_;

  // Whether this particular propagator parametrizes on the tree structure. This
  // is used to speed up the forward propagation to some extent.
  bool param_on_tree;
  bool has_unfolding;

  // D = weights
  // D_grad = gradient
  // Delta_D = Delta (for backprop)
  WeightVectorsType D;
  WeightVectorsType Dr;

  WeightVectorsType D_grad;

  WeightVectorsType Delta_D;
  WeightVectorsType Delta_Dr;

  // Storage for embeddings.
  Real* m_data; // tree storage (embeddings, gradients, deltas ..).
  Real* w_data; // store gradients for actual words in here.
  Real* g_data; // weight matrix and bias gradients.
  int m_data_size;
  int w_data_size;
  int g_data_size;

  friend class BackpropagatorBase;
};

#endif  // COMMON_SINGLEPROPBASE_H
