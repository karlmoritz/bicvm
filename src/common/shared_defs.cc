// File: shared_defs.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 09-05-2013
// Last Update: Wed 14 May 2014 11:17:38 BST

#include "shared_defs.h"
#include "recursive_autoencoder.h"

Model::Model() :
  rae(nullptr), alpha(0.2), beta(0.5), gamma(0.1), normalization_type(0),
  b(nullptr), a(nullptr), docmod(nullptr), it_count(0), num_noise_samples(2),
  noise_sample_offset(1), calc_L2(true) {
    max_sent_length = 0;
    max_node_length = 0;
  }

void Model::finalize() {
  indexes.reserve(corpus.size());
  for (int i = 0; i < int(corpus.size()); ++i) {
    indexes.push_back(i);
    if (corpus[i].words.size() > max_sent_length)
      max_sent_length = corpus[i].words.size();
    if (corpus[i].nodes.size() > max_node_length)
      max_node_length = corpus[i].nodes.size();
  }
}

Model::Model(RecursiveAutoencoderBase* rae_, TrainingCorpus corp) :
  corpus(corp), rae(rae_), alpha(0.2), beta(0.5), gamma(0.1),
  normalization_type(0), b(nullptr), a(nullptr), it_count(0),
  num_noise_samples(2), noise_sample_offset(1), calc_L2(true) {

    max_sent_length = 0;
    max_node_length = 0;
    indexes.reserve(corp.size());
    for (int i = 0; i < int(corp.size()); ++i) {
      indexes.push_back(i);
      if (corp[i].words.size() > max_sent_length)
        max_sent_length = corp[i].words.size();
      if (corp[i].nodes.size() > max_node_length)
        max_node_length = corp[i].nodes.size();
    }
  }

BProps::BProps(Model& a) {
  propA = a.rae->getBackpropagator(a,a.rae->getThetaPlusDictSize());
  if (a.b != nullptr) {
    propB = (*a.b).rae->getBackpropagator(*a.b,a.b->rae->getThetaPlusDictSize());
    if (a.docmod != nullptr) {
      docprop = new BProps(*a.docmod, *(a.b->docmod));
    } else {
      docprop = nullptr;
    }
  }
}
BProps::BProps(Model& a, Model& b) {
  propA = a.rae->getBackpropagator(a,a.rae->getThetaPlusDictSize());
  propB = b.rae->getBackpropagator(b,b.rae->getThetaPlusDictSize());
  docprop = nullptr;
}
BProps::BProps(Model& a, Model& b, Model& c, Model& d) {
  propA = a.rae->getBackpropagator(a,a.rae->getThetaPlusDictSize());
  propB = b.rae->getBackpropagator(b,b.rae->getThetaPlusDictSize());
  docprop = new BProps(c,d);
}
