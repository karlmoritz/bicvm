// File: finetune_classifier.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 29-01-2013
// Last Update: Mon 12 May 2014 16:20:31 BST
/*------------------------------------------------------------------------
 * Description: Additional logistic regression training on the top part
 * of the model (i.e. after the RAE training).
 * Input is variable (e.g. top-layer plus average of all layers)
 * Output is the number of classes (1 in binary classifier)
 *
 *------------------------------------------------------------------------
 * History:
 * TODO:
 *========================================================================
 */

#ifndef FINETUNE_CLASSIFIER_H_6NDFQBY1
#define FINETUNE_CLASSIFIER_H_6NDFQBY1

// Local
#include "shared_defs.h"
#include "config.h"
//#include "sentsim.h"


class FinetuneClassifier
{
public:
  FinetuneClassifier(RecursiveAutoencoderBase& rae, TrainingCorpus& trainC, TrainingCorpus& testC, Real lambdaF,
    Real alpha, int dynamic_mode, int iterations);

  //FinetuneClassifier(RecursiveAutoencoderBase& rae, SentSim::Corpus& trainC, SentSim::Corpus& testC, int mode);
  ~FinetuneClassifier();

  // Subfunctions for LBFGS and SGD training respectively
  /* void trainLbfgs(LineSearchType linesearch); */
  void trainAdaGrad();
  //void finetune_sgd();

  void evaluate(bool test);

protected:
  /* static int lbfgs_progress_( */
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
      /* ); */

  /* static Real lbfgs_evaluate_( */
      /* void *instance, */
      /* const Real *x, // Variables (theta) */
      /* Real *g,       // Put gradient here */
      /* const int n,              // Number of variables */
      /* const Real step);  // line-search step used in this iteration */

public:

  Real finetuneCostAndGrad_(
      // const Real *x,
      Real *g,
      const int n);


private:


  Real*   trainI_;
  Real*   testI_;
  Real*   theta_;

  struct VectorLabelPair {
    WeightVectorType vector;
    int              label;
    VectorLabelPair(WeightVectorType v, int l) : vector(v), label(l) {}
  };

  vector<size_t> mix;

  vector<VectorLabelPair> trainData;
  vector<VectorLabelPair> testData;

  //WeightMatricesType    Wmat;   // Weight matrices for input to sentiment transform
  //WeightVectorsType     Bmat;

  WeightMatricesType    Wcat;   // Weight matrices for labels
  WeightVectorsType     Bcat;   // Bias weights for labels

  int dynamic_embedding_size;
  size_t test_length;
  size_t train_length;
  int label_width;
  int num_label_types;
  int theta_size_;

  Real lambda;
  Real alpha_rae;
  int mode;

  size_t batch_from;
  size_t batch_to;

  int iterations;

public:
  int it_count; // counts iterations, restart lbfgs if below
  int num_batches;
  Real eta;
  /***************************************************************************
 *                              Serialization                              *
 ***************************************************************************/

  friend class boost::serialization::access;
  template<class Archive>
    void save(Archive& ar, const unsigned version) const {
      ar & theta_size_;
      ar & boost::serialization::make_array(theta_, theta_size_);
    }

  template<class Archive>
    void load(Archive& ar, const unsigned version) {
      delete [] theta_;
      theta_ = new Real[theta_size_];
      ar & boost::serialization::make_array(theta_, theta_size_);
    }
  BOOST_SERIALIZATION_SPLIT_MEMBER()

};


#endif /* end of include guard: FINETUNE_CLASSIFIER_H_6NDFQBY1 */
