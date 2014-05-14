// File: singleprop.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 13-01-2013
// Last Update: Wed 14 May 2014 16:30:24 BST

#include <cmath>

#include "singleprop.h"
#include "../../common/fast_math.h"

namespace additive {

SingleProp::SingleProp(RecursiveAutoencoderBase* rae,
                       int max_sent_length, int max_node_length,
                       Real beta = 1, Bools updates = Bools())
  : SinglePropBase(updates, beta, true, false),
  grad_D(0, 0, 0), grad_Wl(0, 0), grad_Bl(0, 0) {

    rae_ = static_cast<RecursiveAutoencoder*>(rae);
    word_width = rae_->config.word_representation_size;

    /***************************************************************************
     *      Create data fields for temporary storage in this propagation       *
     ***************************************************************************/

    m_data_size = 1 * ( // additive model: only one hidden node.
        2 * word_width                    // D, delta_D
        );
    m_data = new Real[m_data_size];
    Real* ptr = m_data;
    for (auto i = 0; i < 1; ++i) {
      D.push_back(WeightVectorType(ptr, word_width));
      ptr += word_width;
      Delta_D.push_back(WeightVectorType(ptr, word_width));
      ptr += word_width;
    }
    assert(ptr == m_data+m_data_size);

    /***************************************************************************
     *             Create data fields for word embedding gradients             *
     ***************************************************************************/

    /***************************************************************************
     *            Create data fields for weight and bias gradients             *
     ***************************************************************************/
  }

/*
 * Reset the temporary variables - in our case here just the single value for
 * the additive bag of words parent node.
 */
void SingleProp::loadWithSentence(const Sentence &t) {
  // memset(m_data, 0, sizeof(Real) * m_data_size);
  SinglePropBase::loadWithSentence(t);
  D[0].setZero();
  Delta_D[0].setZero();
  // D[i] = rae_->de_->D.row(instance_->words[node]);
}

void SingleProp::passDataLink(Real* data, int size) {
    int word_width = rae_->config.word_representation_size;
    int dict_size  = rae_->getDictSize();
    Real *ptr = data;
    new (&grad_D) WeightMatrixType(ptr, dict_size, word_width);
    // grad_D.setZero();
    ptr += rae_->getThetaDSize();
    new (&grad_Wl) WeightVectorType(ptr, rae_->theta_Wl_size_);
    // grad_Wl.setZero();
    ptr += rae_->theta_Wl_size_;
    new (&grad_Bl) WeightVectorType(ptr, rae_->theta_Bl_size_);
    // grad_Bl.setZero();
    ptr += rae_->theta_Bl_size_;
    assert (data + size == ptr);
}

/******************************************************************************
 *                        Actual composition functions                        *
 ******************************************************************************/
void SingleProp::forwardPropagate(bool autoencode) {
  for (int i = 0; i < sent_length; ++i) {
    D[0] += rae_->de_->getD().row(instance_->words[i]);
  }
}

int SingleProp::backPropagate(bool lbl_error,
                              bool rae_error,
                              bool bi_error,
                              bool unf_error,
                              VectorReal *word) {
  // if label error: only on root node
  if (lbl_error)  applyLabel(0, true, 1.0);
  if (bi_error)  backpropBi(0, word);

  if (updates.D) {
    for (int i = 0; i < sent_length; ++i) {
      grad_D.row(instance_->words[i]) += Delta_D[0];
    }
  }
  return 0;
}

void SingleProp::encodeInputs(int node, int child0, int child1, int rule,
                              int rc0, int rc1, bool autoencode) {
  assert(false);
}

void SingleProp::encodeSingular(int node, int child0, int rule, int rc0,
                                bool autoencode) {
  assert(false);
}

int SingleProp::applyLabel(int node, bool use_lbl_error, Real beta) {
  if (node > 0) { std::cerr << "Additive models shouldn't label subtrees" << std::endl; }
  assert(node == 0);
  ArrayReal label_pred = (
      rae_->Wl[0] * D[node] + rae_->Bl[0]
      ).unaryExpr(std::ptr_fun(getSigmoid)).array();

  ArrayReal label_correct(rae_->config.label_class_size);
  assert(rae_->config.label_class_size == 1);
  label_correct[0] = instance_->value;

  // dE/dv * sigmoid'(net)
  ArrayReal label_delta = - beta * (label_correct - label_pred) * (1-label_pred) * (label_pred);
  Real lbl_error = 0.5 * beta * ((label_pred - label_correct) * (label_pred - label_correct)).sum();

  int correct = 0;
  if( abs(instance_->value - label_pred[0]) < 0.5 ) {
    correct = 1;
    classified_correctly_ += 1;
  } else {
    classified_wrongly_ += 1;
  }

  grad_Wl += rae_->alpha_lbl * label_delta.matrix() * D[node].transpose();
  grad_Bl += rae_->alpha_lbl * label_delta.matrix();
  Delta_D[node] += rae_->alpha_lbl * rae_->Wl[0].transpose() * label_delta.matrix();
  lbl_error_ += rae_->alpha_lbl * lbl_error;

  return correct;
}

/******************************************************************************
 *                      Actual backpropagation functions                      *
 ******************************************************************************/

void SingleProp::backpropInputs(int node, int child0, int child1, int rule,
                                int rc0, int rc1, bool rae_error,
                                bool unf_error) {
  assert(false);
}

void SingleProp::backpropWord(int node, int sent_pos) {
  assert(false);
}

void SingleProp::backpropBi(int node, VectorReal* word) {
  ArrayReal label_correct = word->array();
  ArrayReal label_pred = D[node].array();

  // Error: 0.5 (me - other)^2
  ArrayReal label_delta = (label_pred - label_correct);
  Real lbl_error = 0.5 * 0.5 * ((label_pred - label_correct)
                                * (label_pred - label_correct)).sum();
  // Error divided by 0.5 again as we only propagate one side of the equation

  Delta_D[node] += rae_->alpha_lbl * label_delta.matrix();
  lbl_error_ +=  rae_->alpha_lbl * lbl_error;
}

}  // namespace additive
