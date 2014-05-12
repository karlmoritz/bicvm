// File: config.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 02-01-2013
// Last Update: Thu 17 Oct 2013 03:10:44 PM BST
/*------------------------------------------------------------------------
 * Description: Adapted from Phil's config.h for oxlm
 *
 *------------------------------------------------------------------------
 * History:
 * TODO:
 *========================================================================
 */

#ifndef CONFIG_H_DB6BKMJ2
#define CONFIG_H_DB6BKMJ2

#include <iostream>
#include <fstream>
#include <boost/shared_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "shared_defs.h"

struct ModelData {
  enum ZType { Sampled, Exact };

  ModelData() : history(), tree(TREE_FOREST), ztype(Sampled), step_size(Real(0.1)), eta_t0(Real(1.0)),
  l2_parameter(Real(0.0)), time_series_parameter(Real(1.0)), label_sample_size(100),
  feature_type("explicit"), hash_bits(16), threads(1), iteration_size(1),
  verbose(false), ngram_order(3), word_representation_size(10),
  label_class_size(1), uniform(false), num_sentences(1), training_method(0),
  num_weight_types(1), num_label_types(1), approx_width(3), hinge_loss_margin(Real(1.0)),
  model_out("model"), dump_freq(10), init_to_I(false),
  calc_rae(false), calc_lbl(true), calc_bi(false), calc_through(false), calc_uae(false),
  cycles_so_far(0)
  {}

  std::string history;
  TreeType    tree;
  ZType       ztype;
  Real       step_size;
  Real       eta_t0;
  Real       l2_parameter;
  Real       time_series_parameter;
  int         label_sample_size;
  std::string feature_type;
  int         hash_bits;
  int         threads;
  int         iteration_size;
  bool        verbose;
  int         ngram_order;
  int         word_representation_size;
  int         label_class_size;
  bool        uniform;
  int         num_sentences;
  int         training_method;
  int         num_weight_types; // W1in, W2in, W1rec, W2rec for the autoencoder case
  int         num_label_types; // Only one output classifier
  int         approx_width; // width of UV matrices
  Real       hinge_loss_margin; // margin size of hinge loss

  // Temporary variables not to be stored or serialized
  std::string model_out;
  int         dump_freq;
  bool        init_to_I;
  bool        calc_rae;
  bool        calc_lbl;
  bool        calc_bi;
  bool        calc_through;
  bool        calc_uae;
  int         cycles_so_far;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & history;
    ar & tree;
    ar & ztype;
    ar & step_size;
    ar & eta_t0;
    ar & l2_parameter;
    ar & time_series_parameter;
    ar & label_sample_size;
    ar & feature_type;
    ar & hash_bits;
    ar & threads;
    ar & iteration_size;
    ar & verbose;
    ar & ngram_order;
    ar & word_representation_size;
    ar & label_class_size;
    ar & uniform;
    ar & num_sentences;
    ar & training_method;
    ar & num_weight_types;
    ar & num_label_types;
    ar & approx_width;
    ar & hinge_loss_margin;
  }
};
typedef boost::shared_ptr<ModelData> ModelDataPtr;

#endif /* end of include guard: CONFIG_H_DB6BKMJ2 */

