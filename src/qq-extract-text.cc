// File: qq-extract-text.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 01-01-2013
// Last Update: Mon 26 May 2014 16:59:24 BST

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
#include "common/load_plain.h"

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
  cerr << "QQ question and vocab representation extractor: Copyright 2014 Karl Moritz Hermann" << endl;

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
     "type of model (ccaeb, mvrnn, ...)")
    ("input", bpo::value<string>()->default_value(""),
     "input file in paralex vocabulary format")
    ("model", bpo::value<string>(), "input model")
    ("prefix", bpo::value<string>()->default_value(""),
     "prefix for output files <pre>.ent and <pre>.rel")
    ("embeddings", bpo::value<int>()->default_value(-1),
     "use embeddings for baseline dictionary (0=senna,1=turian,2=cldc-en,3=cldc-de)")
    ("tab", bpo::value<bool>()->default_value(false),
     "tab-separate output and add header row")
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

  Model modelA;
  int embeddings = vm["embeddings"].as<int>();
  string inputA = vm["input"].as<string>();
  bool create_new_dict = false;

  Senna sennaA(*deA,embeddings);

  assert(!inputA.empty());
  load_plain::load_file(modelA.corpus, inputA, 0, create_new_dict, sennaA);

  modelA.rae = &raeA;
  modelA.rae->de_ = deA;

  /***************************************************************************
   *    Propagate inputs and store in appropriate output files               *
   ***************************************************************************/

  int asize = modelA.rae->getThetaPlusDictSize();
  BackpropagatorBase* backprop = modelA.rae->getBackpropagator(modelA, asize);

  int word_width = modelA.rae->config.word_representation_size;
  fs::path output_file_path(vm["prefix"].as<string>());
  fs::ofstream output_file(output_file_path);
  {
    // Forward propagate all sentences and store their resulting vector in
    // the document model dictionary.
    string separator = " ";
    if (vm["tab"].as<bool>()) {
      separator = "\t";
      output_file << "Name";
      for (int j = 0; j < word_width; ++j) output_file << "\t" << j;
      output_file << endl;
    }
    VectorReal x(word_width);
    for (int j = 0; j < modelA.corpus.size(); ++j) {
      backprop->forwardPropagate(j,&x);
      output_file << x[0];
      for (int i = 1; i < word_width; ++i) { output_file << separator << x[i]; }
      output_file << endl;
    }
  }
  delete backprop;
  output_file.close();
}
