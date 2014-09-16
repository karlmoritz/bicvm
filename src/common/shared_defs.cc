// File: shared_defs.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 09-05-2013
// Last Update: Mon 15 Sep 2014 14:22:49 BST

#include "shared_defs.h"
#include "recursive_autoencoder.h"
#include "backpropagatorbase.h"

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
    if (corpus.words[i].size() > max_sent_length)
      max_sent_length = corpus.words[i].size();
  }
  for (int i = 0; i < int(corpus.nodes.size()); ++i) {
    if (corpus.nodes[i].size() > max_node_length)
      max_node_length = corpus.nodes[i].size();
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
      if (corp.words[i].size() > max_sent_length)
        max_sent_length = corp.words[i].size();
      if (corp.nodes[i].size() > max_node_length)
        max_node_length = corp.nodes[i].size();
    }
  }

BProps::BProps(const Model& a) {
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

BProps::BProps(const Model& a, const Model& b) {
  propA = a.rae->getBackpropagator(a,a.rae->getThetaPlusDictSize());
  propB = b.rae->getBackpropagator(b,b.rae->getThetaPlusDictSize());
  docprop = nullptr;
}

BProps::BProps(const Model& a, const Model& b, const Model& c, const Model& d) {
  propA = a.rae->getBackpropagator(a,a.rae->getThetaPlusDictSize());
  propB = b.rae->getBackpropagator(b,b.rae->getThetaPlusDictSize());
  docprop = new BProps(c,d);
}

BProps::BProps(const Model& a, bool share_dict) {
  if (share_dict == false) {
    propA = a.rae->getBackpropagator(a,a.rae->getThetaPlusDictSize());
    if (a.b != nullptr) {
      propB = (*a.b).rae->getBackpropagator(*a.b,a.b->rae->getThetaPlusDictSize());
      if (a.docmod != nullptr) {
        docprop = new BProps(*a.docmod, *(a.b->docmod));
      } else {
        docprop = nullptr;
      }
    }
  } else {
    propA = a.rae->getBackpropagator(a,a.rae->getThetaPlusDictSize());
    Real* dictptr = propA->getDataLink();
    propB = (*a.b).rae->getBackpropagator(*a.b,a.b->rae->getThetaSize(),dictptr);
    if (a.docmod != nullptr) {
      docprop = new BProps(*a.docmod, *(a.b->docmod), dictptr);
    } else {
      docprop = nullptr;
    }
  }
}

BProps::BProps(const Model& a, const Model& b, Real* dictptr) {
  propA = a.rae->getBackpropagator(a,a.rae->getThetaSize(), dictptr);
  propB = b.rae->getBackpropagator(b,b.rae->getThetaSize(), dictptr);
  docprop = nullptr;
}
