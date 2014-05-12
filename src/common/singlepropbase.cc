// File: singlepropbase.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 16-10-2013
// Last Update: Mon 06 Jan 2014 05:47:16 PM GMT

#include "singlepropbase.h"

SinglePropBase::SinglePropBase (Bools updates,
                                Real beta, bool param_on_tree, bool has_unfolding)
  : updates(updates), beta_(beta), param_on_tree(param_on_tree),
  has_unfolding(has_unfolding), m_data_size(0), w_data_size(0), g_data_size(0) {
  }

SinglePropBase::~SinglePropBase() {
  delete [] g_data;
  delete [] m_data;
  delete [] w_data;
}

void SinglePropBase::loadWithSentence(const Sentence& t) {
  instance_ = &t; //static_cast<const Sentence&>(t);
  sent_length = int(instance_->words.size());
  nodes_length = int(instance_->rule.size());
  lbl_error_ = 0.0;
  rae_error_ = 0.0;
  classified_correctly_ = 0;
  classified_wrongly_ = 0;
}

void SinglePropBase::forwardPropagate(bool autoencode) {
  for (int i = nodes_length-1; i>= 0; --i)
  {
    int child0 = instance_->child0[i];
    int child1 = instance_->child1[i];
    int rule = instance_->rule[i];
    if(child1 >= 0)
    {
      int rc0 = (param_on_tree) ? instance_->cat[child0] : -1;
      int rc1 = (param_on_tree) ? instance_->cat[child1] : -1;
      encodeInputs(i, child0, child1, rule, rc0, rc1, autoencode);
    }
    else if (child0 >= 0)
    {
      int rc0 = (param_on_tree) ? instance_->cat[child0] : -1;
      encodeSingular(i, child0, rule, rc0, autoencode);
    }
  }
}

int SinglePropBase::backPropagate(bool lbl_error,
                                  bool rae_error,
                                  bool bi_error,
                                  bool unf_error,
                                  VectorReal* word) {
  for (int i = 0; i<nodes_length; ++i)
  {
    int child0 = instance_->child0[i];
    int child1 = instance_->child1[i];
    int rule = instance_->rule[i];
    int rc0 = (param_on_tree && child0 >= 0) ? instance_->cat[child0] : -1;
    int rc1 = (param_on_tree && child1 >= 0) ? instance_->cat[child1] : -1;

    // Get the parallel error (currently only at the top node)
    if (i == 0 && bi_error == true) backpropBi(i, word);
    if (lbl_error) {
      if (child0 == -1) applyLabel(i,true,1.0);
      else              applyLabel(i,true,beta_);
    }

    if (child1 == -1) {
      child1 = child0;
      rc1 = 0;
    }
    if (rule == LEAF) {
      if (updates.D)  backpropWord(i, instance_->nodes[i]);
    } else {
      backpropInputs(i, child0, child1, rule, rc0, rc1, rae_error, unf_error);
    }
  }
  return (classified_correctly_ > classified_wrongly_) ? 1 : 0;
}

void SinglePropBase::unfoldFromHere(int node)
{
  if(not has_unfolding) { throw "Not Implemented"; }

  // Reset all the unfolding variables to zero.
  for (int i=0; i<nodes_length; ++i) {
    Delta_Dr[i].setZero();
  }
  // Set own values in mirror.
  Dr[node] = D[node];

  // Propagate unfolding: from "here" unfold until the input layer.
  queue<int> q;
  stack<int> backq;
  q.push(node); backq.push(node);
  while (not q.empty())
  {
    int i = q.front();
    q.pop();
    int rule = instance_->rule[i];
    int child0 = instance_->child0[i];
    int child1 = instance_->child1[i];
    if (child0 != -1) { q.push(child0); backq.push(child0); }
    if (child1 != -1 and child1 != child0) { q.push(child1); backq.push(child1); }
    if (child1 == -1) { child1 = child0; }
    int rc0 = (param_on_tree && child0 >= 0) ? instance_->cat[child0] : -1;
    int rc1 = (param_on_tree && child1 >= 0) ? instance_->cat[child1] : -1;
    if (rule != LEAF)  unfoldProp(i, child0, child1, rule, rc0, rc1);
  }

  // Backpropagate unfolding: from the reconstructed input back to "here".
  while (not backq.empty())
  {
    int i = backq.top();
    backq.pop();
    int rule = instance_->rule[i];
    int child0 = instance_->child0[i];
    int child1 = instance_->child1[i];
    if (child1 == -1)  child1 = child0;
    int rc0 = (param_on_tree && child0 >= 0) ? instance_->cat[child0] : -1;
    int rc1 = (param_on_tree && child1 >= 0) ? instance_->cat[child1] : -1;
    if (rule == LEAF)
      unfoldRecError(i);
    else
      unfoldBackprop(i, child0, child1, rule, rc0, rc1);
  }

  // Set deltas back into the real thing
  Delta_D[node] += Delta_Dr[node];

}

void SinglePropBase::backpropAllWords() {
  for (int i = 0; i < nodes_length; ++i) {
    if (instance_->rule[i] == LEAF and updates.D) {
      backpropWord(i, instance_->nodes[i]);
    }
  }
}

int SinglePropBase::evaluateSentence() {
  for (int i = 0; i < nodes_length; ++i) {
    int child0 = instance_->child0[i];
    if (child0 == -1) applyLabel(i,true,1.0);
    else applyLabel(i,true,beta_);
  }
  return (classified_correctly_ > classified_wrongly_) ? 1 : 0;
}

void SinglePropBase::setToD(VectorReal* x, int i) { *x = D[i]; }

int SinglePropBase::getSentLength() { return sent_length; }
int SinglePropBase::getNodesLength() { return nodes_length; }
Real SinglePropBase::getLblError() { return lbl_error_; }
Real SinglePropBase::getRaeError() { return rae_error_; }
int SinglePropBase::getJointNodes() { return nodes_length; }
int SinglePropBase::getClassCorrect() { return classified_correctly_; }

void SinglePropBase::setDynamic(WeightVectorType& dynamic, int mode) {
  int shift = 0;
  if(mode >= 10)
  {
    shift = word_width;
    mode -= 10;
  }
  switch (mode)
  {
    case 0:
      // Set first half to the top vector, second average all nodes
      shift *= 2;
      dynamic.segment(shift,word_width) = D[0];
      for (auto i=0; i<nodes_length; ++i)
        dynamic.segment(shift+word_width,word_width) += D[i];
      dynamic.segment(shift+word_width,word_width) /= nodes_length;
      break;
    case 1:
      dynamic.segment(shift,word_width) = D[0];
      break;
    case 2:
      for (auto i=0; i<nodes_length; ++i)
        dynamic.segment(shift,word_width) += D[i];
      dynamic.segment(shift,word_width) /= nodes_length;
      break;
    case 3:
      {
        int n = int(D[0].size());
        int pos = 0;
        for (auto i=0; i<nodes_length; ++i)
        {
          dynamic.segment(shift+pos,n) = D[i];
          pos += n;
        }
      }
      break;
    case 4:
      {
        shift *= 4;
        dynamic.segment(shift,word_width) += D[0];
        for (auto i=0; i<nodes_length; ++i)
          dynamic.segment(shift+word_width,word_width) += D[i];
        dynamic.segment(shift+word_width,word_width) /= nodes_length;
        int leaf_count = 0;
        int inner_count = 0;
        for (auto i=0; i<nodes_length; ++i)
        {
          if (instance_->rule[i] == LEAF)
          {
            leaf_count++;
            dynamic.segment(shift+2*word_width,word_width) += D[i];
          }
          else
          {
            inner_count++;
            dynamic.segment(shift+3*word_width,word_width) += D[i];
          }
        }
        dynamic.segment(shift+2*word_width,word_width) /= max(leaf_count,1);
        dynamic.segment(shift+3*word_width,word_width) /= max(inner_count,1);
      }
      break;
    default:
      cout << "Dynamic number does not exist" << endl;
      assert(false);
  }
}
