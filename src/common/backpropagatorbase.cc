// File: backpropagatorbase.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 18-06-2013
// Last Update: Mon 15 Sep 2014 13:42:29 BST

#include <random>

#include "backpropagatorbase.h"

BackpropagatorBase::BackpropagatorBase (RecursiveAutoencoderBase* rae,
                                        const Model &model)
  : rae_(rae), model(model), weights(0,0), dict_weights(0,0), error_(0.0),
  corrupt(0),
  count_nodes_(0), count_words_(0),
  correctly_classified_sent(0), zero_should_be_one(0), zero_should_be_zero(0),
  one_should_be_zero(0), one_should_be_one(0), is_a_zero(0), is_a_one(0),
  rnd_dist(1,3), rnd_gen(rnd_engine, rnd_dist) {
    word_width = rae_->config.word_representation_size;
    dict_size  = rae_->getDictSize();

    singleprop = rae_->getSingleProp(model.max_sent_length,
                                     model.max_node_length,
                                     model.beta, model.bools);
  }

BackpropagatorBase::~BackpropagatorBase () {
  delete singleprop;
}

SinglePropBase* BackpropagatorBase::forwardPropagate(int i, VectorReal* x) {
  singleprop->loadWithSentence(model.corpus, i);
  singleprop->forwardPropagate(false);
  singleprop->setToD(x,0);
  return singleprop;
}

SinglePropBase* BackpropagatorBase::noisyForwardPropagate(int noise, int truth, VectorReal* x) {
  // DANGER: does not check whether noise and truth have the same lengh. This is
  // assumed.
  Corpus corrupt;
  corrupt.words.push_back(model.corpus[noise]);
  // corrupt = model.corpus[noise];
  assert (model.corpus[noise].size() == model.corpus[truth].size());
  // std::random_device rd;
  // std::mt19937 gen(rd());
  for (int i = 0; i < corrupt[0].size(); ++i) {
    if (rnd_gen() == 3) {
      corrupt[0][i] = model.corpus[truth][i];
    }
  }
  singleprop->loadWithSentence(corrupt, 0);
  singleprop->forwardPropagate(false);
  singleprop->setToD(x,0);
  return singleprop;
}

void BackpropagatorBase::backPropagateRae(int i, VectorReal* x) {
  backPropagateAEWrapper(i, false, x);
}
void BackpropagatorBase::backPropagateUnf(int i, VectorReal* x) {
  backPropagateAEWrapper(i, true, x);
}

void BackpropagatorBase::backPropagateAEWrapper(int i, bool unfold,
                                                VectorReal* x) {
  singleprop->loadWithSentence(model.corpus, i);
  singleprop->forwardPropagate(true);
  singleprop->setToD(x,0);
  if (unfold)
    singleprop->backPropagate(false, false, false, true);
  else
    singleprop->backPropagate(false, true, false, false);

  error_ += singleprop->getRaeError();
  count_words_ += singleprop->getSentLength();
  count_nodes_ += singleprop->getNodesLength();
  collectGradients(singleprop, i);
}

int BackpropagatorBase::backPropagateLbl(int i, VectorReal* x) {
  singleprop->loadWithSentence(model.corpus, i);
  singleprop->forwardPropagate(false);
  singleprop->setToD(x,0);
  int correct = singleprop->backPropagate(true, false, false, false);

    if(model.corpus.value[i]==0) { // Assuming corpus has a label (value)
      is_a_zero += singleprop->getJointNodes();
      zero_should_be_zero += singleprop->getClassCorrect();
      one_should_be_zero += (singleprop->getJointNodes() - singleprop->getClassCorrect());
    } else {
      is_a_one += singleprop->getJointNodes();
      one_should_be_one += singleprop->getClassCorrect();
      zero_should_be_one += (singleprop->getJointNodes() - singleprop->getClassCorrect());
    }
    error_ += singleprop->getLblError();
    correctly_classified_sent += correct;
    count_words_ += singleprop->getSentLength();
    count_nodes_ += singleprop->getNodesLength();
    collectGradients(singleprop, i);
  return correct;
}

void BackpropagatorBase::backPropagateBi(int i, VectorReal* x,
                                         VectorReal word) {
  singleprop->loadWithSentence(model.corpus, i);
  singleprop->forwardPropagate(false);
  singleprop->setToD(x,0);
  singleprop->backPropagate(false, false, true, false, &word);

    error_ += singleprop->getLblError(); //yes, correct. ugly overload
    count_words_ += singleprop->getSentLength();
    count_nodes_ += singleprop->getNodesLength();
    collectGradients(singleprop, i);
}

void BackpropagatorBase::unfoldPropagateGiven(int i, SinglePropBase* otherModel,
                                              VectorReal* gradient) {
  singleprop->loadWithSentence(model.corpus, i);
  singleprop->D[0] = otherModel->D[0];
  singleprop->unfoldFromHere(0); // unfold from root node
  // Manually copy over word vectors due to the direct unfolding call.
  singleprop->backpropAllWords();
  *gradient = singleprop->Delta_D[0];

    error_ += singleprop->getRaeError(); //yes, correct. ugly overload
    collectGradients(singleprop, i);
}

void BackpropagatorBase::backPropagateGiven(int i, SinglePropBase* thisModel,
                                            const VectorReal& gradient) {
  thisModel->Delta_D[0] = gradient;
  thisModel->backPropagate(false, false, false, false); // no lbl, rae, bi, unf
}

void BackpropagatorBase::addCountsAndGradsForGiven(int i,
                                                   SinglePropBase* thisModel) {
    count_words_ += thisModel->getSentLength();
    count_nodes_ += thisModel->getNodesLength();
    collectGradients(thisModel, i);
}

void BackpropagatorBase::printInfo() {
  cout << "  " << correctly_classified_sent << "/" << model.to - model.from << " ";
  cout << "Z: " << is_a_zero << ": (" << zero_should_be_zero << " / " << one_should_be_zero << ")";
  cout << "O: " << is_a_one << ": (" << one_should_be_one << " / " << zero_should_be_one << ")";
  cout << endl;
}

WeightVectorType BackpropagatorBase::dumpWeights() { return weights; }
WeightVectorType BackpropagatorBase::dumpDict() { return dict_weights; }
Real BackpropagatorBase::getError() { return error_; }

void BackpropagatorBase::addError(Real i) {
  error_ += i;
}
