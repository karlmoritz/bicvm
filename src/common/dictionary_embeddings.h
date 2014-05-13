// File: dictionary_embeddings.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 02-01-2013
// Last Update: Tue 13 May 2014 16:37:57 BST

#ifndef COMMON_DICTIONARY_EMBEDDINGS_H
#define COMMON_DICTIONARY_EMBEDDINGS_H

#include <map>

#include "common/shared_defs.h"
#include "common/config.h"
#include "common/dictionary.h"
#include "common/senna.h"
#include "common/reindex_dict.h"

class DictionaryEmbeddings {
 public:
  DictionaryEmbeddings(const ModelData& config);
  void init(bool init_words, bool create_new_theta = true) = 0;
  void createTheta(bool random_init = true);
  void moveToAddress(Real* new_address);

  void initFromDict(const DictionaryEmbeddings& rae,
                    std::map<LabelID, LabelID> n2o_map);

  // Average unknown word by all words.
  void averageUnknownWord();

  Real getLambdaCost(Bools bl, Lambdas lambdas);
  void addLambdaGrad(Real* theta_data, Bools bl, Lambdas lambdas);

  /*
   * Inline functions.
   */
  int getThetaSize() { return theta_D_size_; };

  // Underlying dictionary access functions
  const Dictionary& getDictionary() const { return dict_; }
  Dictionary& getDictionary() { return dict_; }
  int getDictSize() { return dict_.num_labels(); }

  // Maybe private?
  Dictionary dict_;

 protected:
  Real*                 theta_; // data storage
  WeightVectorType      Theta;  // data vector

  WeightMatrixType      D;  // Word embeddings (|v|*d)
  int                   theta_size_; // size of data (not vocabulary size!)

 public:
  friend class Senna; // can create and modify the dictionary.
  friend RecursiveAutoencoderBase* reindex_dict(RecursiveAutoencoderBase& rae,
                                                TrainingCorpus& trainC,
                                                TrainingCorpus& testC);
  friend class boost::serialization::access;

  template<class Archive>
      void save(Archive& ar, const unsigned version) const {
        ar & dict_;
        ar & boost::serialization::make_array(theta_, theta_size_);
        assert(version == 0);
      }

  template<class Archive>
      void load(Archive& ar, const unsigned version) {
        ar & dict_;
        init(false,true);  // initialize arrays and make space for theta_.
        ar & boost::serialization::make_array(theta_, theta_size_);
        assert(version == 0);
      }
  BOOST_SERIALIZATION_SPLIT_MEMBER()
};

#endif  // COMMON_DICTIONARY_EMBEDDINGS_H
