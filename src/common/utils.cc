// File: utils.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 30-01-2013
// Last Update: Mon 15 Sep 2014 14:11:32 BST

#include "utils.h"

// STL
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

// Boost
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

// Local
#include "recursive_autoencoder.h"
#include "singlepropbase.h"
#include "backpropagatorbase.h"
#include "recursive_autoencoder.h"


void dumpModel(Model& model, int k)
{
  {
  std::stringstream fname;
  k = k + model.rae->config.cycles_so_far;
  fname << model.rae->config.model_out << "_i" << k;
  std::ofstream ofs(fname.str());
  boost::archive::text_oarchive oa(ofs);
  oa << *(model.rae) << *(model.rae->de_);
  }

  if (model.b != nullptr)
  {
    std::stringstream fname;
    k = k + model.b->rae->config.cycles_so_far;
    fname << model.b->rae->config.model_out << "_i" << k;
    std::ofstream ofs(fname.str());
    boost::archive::text_oarchive oa(ofs);
    oa << *(model.b->rae) << *(model.b->rae->de_);
  }
}

void printSentence(const Dictionary& dict, const Corpus &c, int sent) {
  for (size_t i = 0; i < c[sent].size() ; ++i) {
    cout << dict.label(c[sent][i]) << " ";
  }
  cout << endl;
}

void paraphraseTest(Model& modelA, int k)
{
  bool useB;
  if (modelA.b != nullptr)
    useB = true;
  else if (modelA.a != nullptr)
    useB = false;
  else
  { cout << "Cannot paraphrase" << endl; return; }

  Model& modelB = (useB == true) ? *modelA.b : *modelA.a;

  int length = min(modelA.to,50);

  int multiplier = 1;
  int dmode = 1;
  if (dmode == 0)  multiplier = 2; // root + average
  int dynamic_embedding_size = multiplier * modelA.rae->config.word_representation_size;

  // Storage for fprops
  WeightVectorsType vectorsA;
  WeightVectorsType vectorsB;
  Real* data = new Real[dynamic_embedding_size * (length + length)];
  Real* ptr = data;
  for (auto i=0; i<length; ++i)
  { vectorsA.push_back(WeightVectorType(ptr, dynamic_embedding_size)); ptr += dynamic_embedding_size; }
  for (auto i=0; i<length; ++i)
  { vectorsB.push_back(WeightVectorType(ptr, dynamic_embedding_size)); ptr += dynamic_embedding_size; }

  // For each sentence in both corpora: fprop and store result
// #pragma omp parallel for schedule(dynamic)
  for (auto i = 0; i<length; ++i)
  {
    SinglePropBase* propagator = modelA.rae->getSingleProp(modelA.corpus.words[i].size(),
                                                         modelA.corpus.nodes[i].size(),
                                                         0.5,modelA.bools);
    propagator->loadWithSentence(modelA.corpus, i);
    propagator->forwardPropagate(false);
    propagator->setDynamic(vectorsA[i],dmode);
    delete propagator;
  }
// #pragma omp parallel for schedule(dynamic)
  for (auto i = 0; i<length; ++i)
  {
    SinglePropBase* propagator = modelB.rae->getSingleProp(modelB.corpus.words[i].size(),
                                                         modelB.corpus.nodes[i].size(),
                                                         0.5,modelB.bools);
    propagator->loadWithSentence(modelB.corpus, i);
    propagator->forwardPropagate(false);
    propagator->setDynamic(vectorsB[i],dmode);
    delete propagator;
  }

  // Then: Compare all sentence level representations and pick most similar
  int correct_count = 0;
  int closest = 0;
  Real value;
  Real cos;

  for (auto a = 0; a<length; ++a)
  {
    //value = ((vectorsA[a].transpose() * vectorsB[0]).sum() / (vectorsA[a].norm() * vectorsB[0].norm()));
    int start = max(0,(a-5+length)%length); // where to start
    value = (vectorsA[a] - vectorsB[start]).squaredNorm();
    closest = start;
    if (a%25 == 0)
      cout << "A " << a << " Norm: " << vectorsA[a].squaredNorm() << " 0/1 " << vectorsA[a][0] << " " << vectorsA[a][1] << endl;
    for (auto c = 0; c<10; ++c)
    {
      auto b = (start+c)%length;
      auto xcos = (1.0 * (vectorsA[a].transpose() * vectorsB[b]).sum() / (vectorsA[a].norm() * vectorsB[b].norm()));
      cos = (vectorsA[a] - vectorsB[b]).squaredNorm();
      if (a%25 == 0)
      {
        cout << "B" << b << " Norm: " << vectorsB[b].squaredNorm() << " 0/1 " << vectorsB[b][0] << " " << vectorsB[b][1] ;
        cout << "  cos " << xcos << " eucl " << cos << endl;
      }
      //cout << cos << endl;
      if (cos < value)
      { closest = b; value = cos; }
    }
    cout << "Closest to " << a << " is " << closest << endl;
    if ( a == closest) ++correct_count;
  }
  // If most similar == current n +1 else -1

  cout << "Iteration " << k << endl;
  cout << "Correct: " << correct_count << "/" << length << ": " << (100.0 * correct_count)/length << "%" << endl;
  delete [] data;
}

void printConfig(const bpo::variables_map& vm) {
  /***************************************************************************
   *                   Print brief summary of model setup                    *
   ***************************************************************************/

  cerr << "################################" << endl;
  cerr << "# Config Summary" << endl;
  for (bpo::variables_map::const_iterator iter = vm.begin(); iter != vm.end(); ++iter)
  {
    cerr << "# " << iter->first << " = ";
    const ::std::type_info& type = iter->second.value().type() ;
    if ( type == typeid( ::std::string ) )
      cerr << iter->second.as<string>() << endl;
    if ( type == typeid( int ) )
      cerr << iter->second.as<int>() << endl;
    if ( type == typeid( float ) )
      cerr << iter->second.as<float>() << endl;
    if ( type == typeid( double ) )
      cerr << iter->second.as<double>() << endl;
    if ( type == typeid( bool ) )
      cerr << iter->second.as<bool>() << endl;
  }
  cerr << "################################" << endl;

}
