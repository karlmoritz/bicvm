// File: recursive_autoencoder.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 02-01-2013
// Last Update: Mon 12 May 2014 16:28:07 BST

#ifndef COMMON_RECURSIVE_AUTOENCODER_H
#define COMMON_RECURSIVE_AUTOENCODER_H

/* #include <lbfgs.h> */
#include <map>

#include "common/shared_defs.h"
#include "common/config.h"
#include "common/dictionary.h"
#include "common/senna.h"
#include "common/reindex_dict.h"

class SinglePropBase;
class BackpropagatorBase;

class RecursiveAutoencoderBase {
 public:
  RecursiveAutoencoderBase(const ModelData& config);
  // explicit RecursiveAutoencoderBase(const ModelData& config);
  virtual ~RecursiveAutoencoderBase();
  virtual RecursiveAutoencoderBase* cloneEmpty() = 0;

  // Initializes theta_ / copies theta to new location.
  void finalizeDictionary(bool random_init = true);
  void finalizeSpecific(Real* theta_address);

  void initFromWithDict(const RecursiveAutoencoderBase& rae,
                        std::map<LabelID, LabelID> n2o_map);

  virtual Real getLambdaCost(Bools bl, Lambdas lambdas) = 0;
  virtual void addLambdaGrad(Real* theta_data, Bools bl, Lambdas lambdas) = 0;

  void debugSize(int count);

  // const Dictionary& getDictionary() const { return dict_; }.
  const Dictionary& getDictionary() const { return dict_; }
  Dictionary& getDictionary() { return dict_; }
  int getDictSize() { return dict_.num_labels(); }

  // Average unknown word by all words.
  void averageUnknownWord();

  int getThetaSize();

  virtual void setIncrementalCounts(Counts *counts, Real *&vars,
                                    int &number) = 0;
  // type: rae=0, lbl=1, bi=2
  virtual BackpropagatorBase* getBackpropagator(const Model &model, int n) = 0;
  virtual SinglePropBase* getSingleProp(int sl, int nl, Real beta,
                                        Bools updates) = 0;
  virtual void init(bool init_words, bool create_new_theta = true) = 0;

  ModelData config;
  Dictionary dict_;

  Real alpha_rae;
  Real alpha_lbl;

 protected:
  WeightMatrixType      D;  // Domain    (vector) (nx1)
  WeightVectorType      Theta_Full;
  WeightVectorType      Theta_D;

  Real*                 theta_;
  Real*                 theta_D_;

  int                   theta_size_;
  int                   theta_D_size_;


 public:
  friend class Senna;
  friend void setVarsAndNumber(Real *&vars, int &number_vars, Model &model);
  friend int main(int argc, char **argv);

  friend RecursiveAutoencoderBase* reindex_dict(RecursiveAutoencoderBase& rae,
                                                TrainingCorpus& trainC,
                                                TrainingCorpus& testC);
  friend class boost::serialization::access;
  friend void computeBiCostAndGrad(Model& modelA, Model& modelB, const Real* x,
                                   Real* gradient_location, int n,
                                   int iteration, BProps& prop, Real* error);

  template<class Archive>
      void save(Archive& ar, const unsigned version) const {
        ar & dict_;
        ar & config;
        ar & boost::serialization::make_array(theta_, theta_size_);
        assert(version == 0);
      }

  template<class Archive>
      void load(Archive& ar, const unsigned version) {
        ar & dict_;
        ar & config;
        init(false);  // initialize arrays and make space for theta_.
        ar & boost::serialization::make_array(theta_, theta_size_);
        assert(version == 0);
      }
  BOOST_SERIALIZATION_SPLIT_MEMBER()
};

#endif  // COMMON_RECURSIVE_AUTOENCODER_H
