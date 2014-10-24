// File: qq-rerank-eval.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 01-01-2013
// Last Update: Fri 24 Oct 2014 12:41:40 BST

// STL
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// Boost
#include <boost/algorithm/string.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/foreach.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/regex.hpp>
# if defined(RECENT_BOOST)
#include <boost/locale.hpp>
# endif

// Local
#include "common/shared_defs.h"
#include "common/dictionary.h"
#include "common/config.h"
#include "common/models.h"
#include "common/load_qqpair.h"

#include "common/senna.h"
#include "common/reindex_dict.h"
#include "common/utils.h"

#define EIGEN_DONT_PARALLELIZE

using namespace std;
namespace bpo = boost::program_options;
namespace fs = boost::filesystem;

int main(int argc, char **argv)
{
# if defined(RECENT_BOOST)
  boost::locale::generator gen;
  std::locale l = gen("de_DE.UTF-8");
  std::locale::global(l);
# endif
  cerr << "Paralex question query reranking evaluation: Copyright 2014 Karl Moritz Hermann" << endl;

  /***************************************************************************
   *                         Command line processing                         *
   ***************************************************************************/

  bpo::variables_map vm;

  // Command line processing
  bpo::options_description cmdline_specific("Command line specific options");
  cmdline_specific.add_options()
    ("help,h", "print help message")
    ;
  bpo::options_description generic("Allowed options");
  generic.add_options()
    ("type", bpo::value<string>()->default_value("additive"),
     "type of model (additive, flattree, ...)")
    ("input", bpo::value<string>()->default_value(""),
     "input file in paralex vocabulary format")
    ("model", bpo::value<string>(), "input model")
    ("prefix", bpo::value<string>()->default_value(""),
     "prefix for output files <pre>.results and <pre>.rawdump")
    ("embeddings", bpo::value<int>()->default_value(-1),
     "use embeddings for baseline dictionary (0=senna,1=turian,2=cldc-en,3=cldc-de)")
    ("eas", bpo::value<bool>()->default_value(true),
     "treat entities as string concatenations (true) or as unigram (false)?")
    ;
  bpo::options_description all_options;
  all_options.add(generic).add(cmdline_specific);
  store(parse_command_line(argc, argv, all_options), vm);
  notify(vm);

  if (vm.count("help")) {
    cout << all_options << "\n";
    return 1;
  }

  // Print model config options.
  printConfig(vm);
  ModelData config;

  RecursiveAutoencoderBase* raeptrA = nullptr;

  // Factor out at some stage!
  string type = vm["type"].as<string>();
  if (type == "additive") {
    raeptrA = new additive::RecursiveAutoencoder(config);
  } else if (type == "flattree") {
    raeptrA = new flattree::RecursiveAutoencoder(config);
  } else if (type == "additive_avg") {
    raeptrA = new additive_avg::RecursiveAutoencoder(config);
  } else {
    cout << "Model (" << type << ") does not exist" << endl; return -1;
  }

  RecursiveAutoencoderBase& raeA = *raeptrA;

  /***************************************************************************
   *              Read in model and create raeA and docraeA                   *
   ***************************************************************************/

  assert(vm.count("model")); // Without a model this makes no sense.

  std::ifstream ifs(vm["model"].as<string>());
  boost::archive::text_iarchive ia(ifs);
  ia >> raeA;
  DictionaryEmbeddings* deA = new DictionaryEmbeddings(raeA.config.word_representation_size);
  ia >> *deA;

  Model modelA, modelB, modelC;
  int embeddings = vm["embeddings"].as<int>();
  string inputA = vm["input"].as<string>();
  bool create_new_dict = false;
  bool remove_type_ending = vm["eas"].as<bool>();

  Senna sennaA(*deA,embeddings);

  assert(!inputA.empty());
  load_qqpair::load_file(modelA.corpus, modelB.corpus, modelC.corpus, inputA,
                         create_new_dict, remove_type_ending, sennaA);

  modelA.rae = &raeA;
  modelA.rae->de_ = deA;
  modelB.rae = &raeA;
  modelB.rae->de_ = deA;
  modelC.rae = &raeA;
  modelC.rae->de_ = deA;

  /***************************************************************************
   *            Read in input file and create sentence strings               *
   ***************************************************************************/

  // std::ifstream infile(vm["input"].as<string>());
  // std::string line;
  // std::vector<std::string> sentence_strings;
  // std::vector<std::string> types;
  // while (std::getline(infile, line)) {
    // trim(line);
    // type = line.substr(line.length()-1,1);
    // line = line.substr(line.find(" ")+1);
    // sentence_strings.push_back(line);
    // types.push_back(type);
  // }

  /***************************************************************************
   *    Propagate inputs and store in appropriate output files               *
   ***************************************************************************/

  int asize = modelA.rae->getThetaPlusDictSize();
  int bsize = modelB.rae->getThetaPlusDictSize();
  int csize = modelC.rae->getThetaPlusDictSize();
  BackpropagatorBase* backpropS  = modelA.rae->getBackpropagator(modelA, asize);
  BackpropagatorBase* backpropQ1 = modelB.rae->getBackpropagator(modelB, bsize);
  BackpropagatorBase* backpropQ2 = modelC.rae->getBackpropagator(modelC, csize);

  Real cosineA,cosineB,euclideanA,euclideanB;
  Real cosine,euclidean;

  int word_width = modelA.rae->config.word_representation_size;
  fs::path res_file_path(vm["prefix"].as<string>() + ".cosine");
  fs::path raw_file_path(vm["prefix"].as<string>() + ".euclid");
  fs::ofstream res_file(res_file_path);
  fs::ofstream raw_file(raw_file_path);
  {
    // Forward propagate all sentences and store their resulting vector in
    // the document model dictionary.
    VectorReal x(word_width), queryA(word_width), queryB(word_width);
    for (int j = 0; j < modelA.corpus.size(); ++j) {
      backpropS->forwardPropagate(j,&x);
      backpropQ1->forwardPropagate(j,&queryA);
      backpropQ2->forwardPropagate(j,&queryB);
      // Ensure we only consider queries for which we have both embeddings.
      if (modelB.corpus.words[j][1] != 0) {
        cosineA = ((x.transpose() * queryA).sum() / (x.norm() * queryA.norm()));
        euclideanA = (x - queryA).squaredNorm();
      }
      if (modelC.corpus.words[j][1] != 0) {
        cosineB = ((x.transpose() * queryB).sum() / (x.norm() * queryB.norm()));
        euclideanB = (x - queryB).squaredNorm();
      }
      cosine = (modelB.corpus.words[j][1] != 0) ? max(cosineA,cosineB) : cosineB;
      euclidean = (modelB.corpus.words[j][1] != 0) ? min(euclideanA,euclideanB) : euclideanB;
      res_file << cosineA << endl;
      raw_file << euclidean << endl;
    }
  }
  delete backpropS, backpropQ1, backpropQ2;
  res_file.close();
  raw_file.close();
}
