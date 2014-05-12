// File: finetune_classifier.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 29-01-2013
// Last Update: Mon 12 May 2014 16:21:15 BST
/*------------------------------------------------------------------------
 * Description: <DESC>
 *
 *------------------------------------------------------------------------
 * History:
 * TODO:   Delete things at the end
 *========================================================================
 */


// STL
#include <iostream>
#include <algorithm>
#include <random>
#include <chrono>

// Boost
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>

// Local
#include "finetune_classifier.h"
#include "models.h"
#include "fast_math.h"

using namespace std;
namespace bpo = boost::program_options;

FinetuneClassifier::FinetuneClassifier(RecursiveAutoencoderBase& rae,
    TrainingCorpus& trainC, TrainingCorpus& testC, Real lambdaF, Real alpha,
    int dynamic_mode, int iterations) : lambda(lambdaF), alpha_rae(alpha),
  mode(dynamic_mode), iterations(iterations), it_count(0), num_batches(100),
  eta(Real(0.1)) {

  /***************************************************************************
   *             Define a couple of frequently needed variables              *
   ***************************************************************************/

  train_length = min(rae.config.num_sentences,int(trainC.size()));
  bool use_full_corpus = false;
  if (train_length == 0)
  {
    train_length = trainC.size();
    test_length = testC.size();
    use_full_corpus = true;
  }
  else
    test_length = testC.size(); //train_length;

  label_width = rae.config.label_class_size;
  num_label_types = rae.config.num_label_types; // 1

  int multiplier = 1;
  if (mode == 0)
    multiplier = 2;
  if (mode == 3)
    multiplier = 3; // only works for compound test
  if (mode == 4)
    multiplier = 4;
  // Embedding: s1 + s2 + cos_sim(s1,s2) + len(s1) + len(s2) +
  // unigram_overlap(s1,s2) following Blacoe/Lapata 2012
  dynamic_embedding_size = multiplier * rae.config.word_representation_size;

  theta_size_ = dynamic_embedding_size * label_width * num_label_types + label_width * num_label_types;

  trainI_ = new Real[train_length * dynamic_embedding_size]();
  testI_  = new Real[test_length * dynamic_embedding_size]();
  theta_  = new Real[theta_size_];
  WeightVectorType theta(theta_,theta_size_);
  theta.setZero();
  if (true) {
    std::random_device rd;
    std::mt19937 gen(rd());
    //std::mt19937 gen(0);
    Real r = sqrt( 6.0 / dynamic_embedding_size);
    std::uniform_real_distribution<> dis(-r,r);
    for (int i=0; i<theta_size_; i++)
      theta(i) = dis(gen);
  }

  Real* ptr = trainI_;
  for (size_t i = 0; i < train_length; ++i) {
    size_t j = i;
    if ((not use_full_corpus) and (i % 2 == 1))
      j = trainC.size() - i;
    trainData.push_back(VectorLabelPair(WeightVectorType(ptr, dynamic_embedding_size),trainC[j].value));
    mix.push_back(i);
    ptr += dynamic_embedding_size;
  }
  ptr = testI_;
  for (size_t i = 0; i < test_length; ++i) {
    size_t j = i;
    if ((not use_full_corpus) and (i%2 == 1))
      j = testC.size() - i;
    testData.push_back(VectorLabelPair(WeightVectorType(ptr, dynamic_embedding_size),testC[j].value));
    ptr += dynamic_embedding_size;
  }

  ptr = theta_;
  for (auto i = 0; i < num_label_types; ++i) {
    Wcat.push_back(WeightMatrixType(ptr, label_width, dynamic_embedding_size));
    ptr += label_width * dynamic_embedding_size;
  }
  for (auto i = 0; i < num_label_types; ++i) {
    Bcat.push_back(WeightVectorType(ptr, label_width));
    Bcat.back().setZero(); // discuss ..
    ptr += label_width;
  }

  /***************************************************************************
   *    Populate test and train input with forward Propagation and tricks    *
   ***************************************************************************/

#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < train_length; ++i)
  {
    size_t j = i;
    if ((not use_full_corpus) and (i%2 == 1))
      j = trainC.size() - i;


    // TODO(kmh): This could be much more efficient with a single singleprop
    // shared across the corpus.
    Bools bools;
    SinglePropBase* propagator = rae.getSingleProp(trainC[j].words.size(),
                                                     trainC[j].nodes.size(),
                                                     0.5, bools);
    propagator->loadWithSentence(trainC[j]);

    propagator->forwardPropagate(false);
    propagator->setDynamic(trainData[i].vector,mode);
    //cout << "C: " << trainData[i].vector[0] << endl;

    delete propagator;
  }

#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i<test_length; ++i)
  {
    size_t j = i;
    if ((not use_full_corpus) and (i%2 == 1))
      j = testC.size() - i;

    Bools bools;
    SinglePropBase* propagator = rae.getSingleProp(testC[j].words.size(),
                                                     testC[j].nodes.size(),
                                                     0.5, bools);
    propagator->loadWithSentence(testC[j]);

    propagator->forwardPropagate(false);
    propagator->setDynamic(testData[i].vector,mode);

    delete propagator;
  }
}

/*
 *FinetuneClassifier::FinetuneClassifier(RecursiveAutoencoderBase& rae,
 *    SentSim::Corpus& trainC, SentSim::Corpus& testC, int mode)
 *  : lambda(0.0001), mode(mode), it_count(0), num_batches(100), eta(0.1)
 *{
 *
 *  #***************************************************************************
 *   *                Set the unknown word to the average word                 *
 *   ***************************************************************************#
 *
 *  rae.We.row(0) = rae.We.colwise().sum() / (rae.getDictSize() - 1);
 *
 *  #***************************************************************************
 *   *                        Set some basic variables                         *
 *   ***************************************************************************#
 *
 *
 *  label_width = rae.config.label_class_size;
 *  num_label_types = rae.config.num_label_types; // 1
 *
 *  bool add_combs = false;
 *  if (mode > 10)
 *  {
 *    mode -= 10;
 *    add_combs = true;
 *  }
 *
 *  int multiplier = 1;
 *  if (mode == 0)
 *    multiplier = 2;
 *  if (mode == 3)
 *    multiplier = 3; // only works for compound test
 *  if (mode == 4)
 *    multiplier = 4;
 *
 *  // Embedding: s1 + s2 + cos_sim(s1,s2) + len(s1) + len(s2) +
 *  // unigram_overlap(s1,s2) following Blacoe/Lapata 2012
 *  dynamic_embedding_size = multiplier * rae.config.word_representation_size * 2 + 4;
 *  int single_embedding_size = multiplier * rae.config.word_representation_size;
 *
 *  if (add_combs)
 *    dynamic_embedding_size += 2 * single_embedding_size;
 *
 *
 *  train_length = trainC.size();
 *  test_length = testC.size();
 *
 *  theta_size_ = dynamic_embedding_size * label_width * num_label_types + label_width * num_label_types;
 *  trainI_ = new Real[train_length * dynamic_embedding_size]();
 *  testI_  = new Real[test_length * dynamic_embedding_size]();
 *  theta_  = new Real[theta_size_]();
 *
 *  WeightVectorType theta(theta_,theta_size_);
 *  theta.setZero();
 *
 *  Real* ptr = trainI_;
 *  for (auto i=0; i<train_length; ++i) {
 *    trainData.push_back(VectorLabelPair(WeightVectorType(ptr, dynamic_embedding_size),trainC[i].value));
 *    mix.push_back(i);
 *    ptr += dynamic_embedding_size;
 *  }
 *  ptr = testI_;
 *  for (auto i=0; i<test_length; ++i) {
 *    testData.push_back(VectorLabelPair(WeightVectorType(ptr, dynamic_embedding_size),testC[i].value));
 *    ptr += dynamic_embedding_size;
 *  }
 *
 *  ptr = theta_;
 *  for (auto i=0; i<num_label_types; ++i) {
 *    Wcat.push_back(WeightMatrixType(ptr, label_width, dynamic_embedding_size));
 *    ptr += label_width * dynamic_embedding_size;
 *  }
 *  for (auto i=0; i<num_label_types; ++i) {
 *    Bcat.push_back(WeightVectorType(ptr, label_width));
 *    ptr += label_width;
 *  }
 *
 *  #***************************************************************************
 *   *    Populate test and train input with forward Propagation and tricks    *
 *   ***************************************************************************#
 *
 *
 *#pragma omp parallel for schedule(dynamic)
 *  for (auto i = 0; i<train_length; ++i)
 *  {
 *    SinglePropBase* s1 = rae.getSingleProp(trainC[i].s1,0.0);
 *    SinglePropBase* s2 = rae.getSingleProp(trainC[i].s2,0.0);
 *
 *    assert (trainData[i].vector.sum() == 0);
 *
 *    s1->forwardPropagate(true);
 *    s1->setDynamic(trainData[i].vector,mode);
 *    s2->forwardPropagate(true);
 *    s2->setDynamic(trainData[i].vector,mode+10);
 *
 *    auto one = trainData[i].vector.segment(0,single_embedding_size);
 *    auto two = trainData[i].vector.segment(single_embedding_size,single_embedding_size);
 *    Real cosine_similarity = (one.transpose() * two).sum() / (one.norm() * two.norm());
 *    trainData[i].vector[2*single_embedding_size] = cosine_similarity;
 *    trainData[i].vector[2*single_embedding_size+1] = s1->getSentLength();
 *    trainData[i].vector[2*single_embedding_size+2] = s2->getSentLength();
 *    trainData[i].vector[2*single_embedding_size+3] = trainC[i].overlap;
 *
 *    if (add_combs)
 *    {
 *      trainData[i].vector.segment(2*single_embedding_size+4,single_embedding_size) = one - two;
 *      trainData[i].vector.segment(3*single_embedding_size+4,single_embedding_size) = one.array() * two.array();
 *    }
 *
 *    delete s1;
 *    delete s2;
 *  }
 *
 *#pragma omp parallel for schedule(dynamic)
 *  for (auto i = 0; i<test_length; ++i)
 *  {
 *    SinglePropBase* s1 = rae.getSingleProp(testC[i].s1,0.0);
 *    SinglePropBase* s2 = rae.getSingleProp(testC[i].s2,0.0);
 *
 *    s1->forwardPropagate(true);
 *    s1->setDynamic(testData[i].vector,mode);
 *    s2->forwardPropagate(true);
 *    s2->setDynamic(testData[i].vector,mode+10);
 *
 *    auto one = testData[i].vector.segment(0,single_embedding_size);
 *    auto two = testData[i].vector.segment(single_embedding_size,single_embedding_size);
 *    Real cosine_similarity = (one.transpose() * two).sum()
 *      / (one.norm() * two.norm());
 *    testData[i].vector[2*single_embedding_size] = cosine_similarity;
 *    testData[i].vector[2*single_embedding_size+1] = s1->getSentLength();
 *    testData[i].vector[2*single_embedding_size+2] = s2->getSentLength();
 *    testData[i].vector[2*single_embedding_size+3] = testC[i].overlap;
 *
 *    delete s1;
 *    delete s2;
 *  }
 *}
 */

void FinetuneClassifier::evaluate(bool test)
{

  size_t length = test_length;
  vector<VectorLabelPair>& data = (test) ? testData : trainData;
  if (test)
    cout << "Test:     ";
  else
  {
    cout << "Training: ";
    length = train_length;
  }

  int right = 0;
  int wrong = 0;
  int tp = 0;
  int fp = 0;
  int tn = 0;
  int fn = 0;

#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < length; ++i)
  {

    // Encode input
    ArrayReal label_vec = (
        Wcat[0] * data[i].vector + Bcat[0]
        ).unaryExpr(std::ptr_fun(getSigmoid)).array();

    ArrayReal lbl_sm = data[i].label - label_vec;

#pragma omp critical
    {
      if(abs(lbl_sm.sum()) > 0.5)
      {
        wrong += 1;
        if(data[i].label == 0)  ++fp;
        else                    ++fn;
      }
      else
      {
        right += 1;
        if(data[i].label == 0)  ++tn;
        else                    ++tp;
      }
    }
  }
    cout << right << "/" << right + wrong << "  ";
    Real precision = 1.0 * tp / (tp + fp);
    Real recall    = 1.0 * tp / (tp + fn);
    Real accuracy  = 1.0 * (tp + tn) / (tp + tn + fp + fn);
    Real f1score   = 2.0 * (precision * recall) / (precision + recall);
    cout << "Acc/F1: " << accuracy << " " << f1score << endl;
}

/* void FinetuneClassifier::trainLbfgs(LineSearchType linesearch) */
/* { */
  /* batch_from = 0; */
  /* batch_to = train_length; */

  /* lbfgs_parameter_t param; */
  /* lbfgs_parameter_init(&param); */
  /* param.linesearch = linesearch; */
  /* param.max_iterations = iterations; */
  /* //param.epsilon = 0.00000001; */
  /* param.m = 25; */

  /* const int n = theta_size_; */
  /* auto vars = theta_; */
  /* Real error = 0.0; */

  /* int tries = 0; */

  /* while (tries < 3 and it_count < 250) */
  /* { */
    /* int ret = lbfgs(n, vars, &error, lbfgs_evaluate_, lbfgs_progress_, this, &param); */
    /* cout << "L-BFGS optimization terminated with status code = " << ret << endl; */
    /* cout << "fx=" << error << endl; */
    /* ++tries; */
  /* } */
/* } */

void FinetuneClassifier::trainAdaGrad()
{
  auto vars = theta_;
  int number_vars = theta_size_;

  WeightArrayType theta(vars,number_vars);

  Real* Gt_d = new Real[number_vars];
  Real* Ginv_d = new Real[number_vars];
  WeightArrayType Gt(Gt_d,number_vars);
  WeightArrayType Ginv(Ginv_d,number_vars);
  Gt.setZero();

  Real* data1 = new Real[number_vars];

  int batchsize = int(train_length / num_batches) + 1;
  cout << "Batch size: " << batchsize << endl;

  for (auto iteration = 0; iteration < iterations; ++iteration)
  {
    //unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    random_shuffle (mix.begin(), mix.end()); //, std::default_random_engine(seed));

    for (auto batch = 0; batch < num_batches; ++batch)
    {
      batch_from = batch*batchsize;
      batch_to = min(size_t((batch+1)*batchsize), train_length);

      if (batch_to - batch_from > 0)
      {
        WeightArrayType grad(data1,number_vars);

        finetuneCostAndGrad_(data1,number_vars); // removed theta_
        Gt += grad*grad;
        for (int i=0;i<number_vars;i++) {
          if (abs(Gt_d[i]) < 0.0000000000000001)
            Ginv_d[i] = 0;
          else
            Ginv_d[i] = sqrt(1/Gt_d[i]);
        }

        grad *= Ginv;
        grad *= eta;

        //cout << theta.abs().sum() << " vs " << grad.abs().sum() << endl;
        theta -= grad;
      }
    }

    evaluate(false);
    evaluate(true);
  }

  delete [] data1;
  delete [] Gt_d;
  delete [] Ginv_d;

}

/* Real FinetuneClassifier::lbfgs_evaluate_( */
    /* void *instance, */
    /* const Real *x, */
    /* Real *g, */
    /* const int n, */
    /* const Real step */
    /* ) */
/* { */
  /* return reinterpret_cast<FinetuneClassifier*>(instance)->finetuneCostAndGrad_(g, n); */
/* } */

/* int FinetuneClassifier::lbfgs_progress_( */
    /* void *instance, */
    /* const Real *x, */
    /* const Real *g, */
    /* const Real fx, */
    /* const Real xnorm, */
    /* const Real gnorm, */
    /* const Real step, */
    /* int n, */
    /* int k, */
    /* int ls */
    /* ) */
/* { */
  /* cout << "N: " << n << endl; */
  /* printf("Iteration %d:\n", k); */
  /* printf("  fx = %f, x[0] = %f, x[1] = %f %f\n", fx, x[0], x[1], x[2]); */
  /* printf("  fx = %f, g[0] = %f, g[1] = %f %f\n", fx, g[0], g[1], g[2]); */
  /* printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step); */
  /* printf("\n"); */


  /* reinterpret_cast<FinetuneClassifier*>(instance)->it_count++; */
  /* reinterpret_cast<FinetuneClassifier*>(instance)->evaluate(true); */
  /* reinterpret_cast<FinetuneClassifier*>(instance)->evaluate(false); */
  /* return 0; */
/* } */

// ForwardPropagates and returns error and gradient on self
Real FinetuneClassifier::finetuneCostAndGrad_(
    // const Real *x,
    Real *gradient_location,
    const int n)
{
  assert(n == theta_size_);

  WeightVectorType  grad(gradient_location,theta_size_);
  grad.setZero();

  WeightMatricesType Wcatgrad;
  WeightVectorsType  Bcatgrad;

  Real* ptr = gradient_location;
  for (auto i=0; i<num_label_types; ++i) {
    Wcatgrad.push_back(WeightMatrixType(ptr, label_width, dynamic_embedding_size));
    ptr += label_width * dynamic_embedding_size;
  }
  for (auto i=0; i<num_label_types; ++i) {
    Bcatgrad.push_back(WeightVectorType(ptr, label_width));
    ptr += label_width;
  }

  assert(ptr == theta_size_ + gradient_location);

  /***************************************************************************
   *    Populate test and train input with forward Propagation and tricks    *
   ***************************************************************************/
  Real cost = 0.0;
  int right = 0;
  int wrong = 0;
  int tp = 0;
  int fp = 0;
  int tn = 0;
  int fn = 0;

#pragma omp parallel for schedule(dynamic)
  for (auto k = batch_from; k<batch_to; ++k)
  {
    auto i = mix[k];

    // Encode input
    ArrayReal label_vec = (
        Wcat[0] * trainData[i].vector + Bcat[0]
        ).unaryExpr(std::ptr_fun(getSigmoid)).array();

    ArrayReal lbl_sm = label_vec - trainData[i].label;
    ArrayReal delta = lbl_sm * (label_vec) * (1 - label_vec);

#pragma omp critical
    {
      //cout << "D/D2" << delta << " " << delta2 << endl;
      cost += 0.5 * (lbl_sm * lbl_sm).sum();
      Wcatgrad[0] += delta.matrix() * trainData[i].vector.transpose();
      Bcatgrad[0] += delta.matrix();

      if(abs(lbl_sm.sum()) > 0.5)
      {
        wrong += 1;
        if(trainData[i].label == 0)
          ++fp;
        else
          ++fn;
      }
      else
      {
        right += 1;
        if(trainData[i].label == 0)
          ++tn;
        else
          ++tp;
      }
    }
  }

  Real lambda_partial = lambda * Real(batch_to - batch_from) / Real(train_length);
  Wcatgrad[0] += lambda_partial*Wcat[0];
  cost += 0.5*lambda_partial*(Wcat[0].cwiseProduct(Wcat[0])).sum();

  Wcatgrad[0] /= Real(batch_to - batch_from);
  cost /= Real(batch_to - batch_from);

  return cost;
}


FinetuneClassifier::~FinetuneClassifier()
{
  delete [] trainI_;
  delete [] testI_;
  delete [] theta_;

}
