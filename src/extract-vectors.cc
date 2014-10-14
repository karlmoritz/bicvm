// File: extract-vectors.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 01-01-2013
// Last Update: Fri 23 May 2014 09:53:54 BST

// STL
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <queue>
#include <vector>

// Boost
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
# if defined(RECENT_BOOST)
#include <boost/locale.hpp>
# endif

// Local
#include "common/shared_defs.h"
#include "common/dictionary.h"

#include "common/config.h"

#include "common/models.h"

#include "common/senna.h"
#include "common/reindex_dict.h"
#include "common/utils.h"

#define EIGEN_DONT_PARALLELIZE

using namespace std;
namespace bpo = boost::program_options;

int main(int argc, char **argv)
{
# if defined(RECENT_BOOST)
  boost::locale::generator gen;
  std::locale l = gen("de_DE.UTF-8");
  std::locale::global(l);
# endif
  cerr << "Extract vectors for words/the whole dictionary: Copyright 2014 Karl Moritz Hermann" << endl;

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
     "type of model (additive, ccaeb, mvrnn, ...)")
    ("input", bpo::value<string>(),
     "vocabulary list for which vectors should be printed")
    ("model,m", bpo::value<string>(), "input model")
    ("output", bpo::value<string>()->default_value(""),
     "vocabulary output file")
    ;
  bpo::options_description all_options;
  all_options.add(generic).add(cmdline_specific);
  store(parse_command_line(argc, argv, all_options), vm);
  notify(vm);

  if (vm.count("help")) {
    cout << all_options << "\n";
    return 1;
  }

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

  int word_width = raeA.config.word_representation_size;

  if (vm.count("input")) {
    // Iterate over some input file.
    string inputA = vm["input"].as<string>();
    Dictionary dictA = deA->getDictionary();
    std::ifstream infile(vm["input"].as<string>());
    std::string line;
    while (std::getline(infile, line)) {
      LabelID id = dictA.id(line);
      if (id != dictA.m_bad_label) {
        cout << line;
        for (int i = 0; i < word_width; ++i) {
          cout << " " << deA->getD()(id, i);
        }
        cout << endl;
      } else {
        cerr << "Not found in dict: " << line << endl;
      }
    }
  } else {
    // Iterate over the dictionary.
    Dictionary dictA = deA->getDictionary();
    for(LabelID id = dictA.min_label(); id < dictA.max_label(); ++id) {
      cout << dictA.label(id);
      for (int i = 0; i < word_width; ++i) {
        cout << " " << deA->getD()(id, i);
      }
      cout << endl;
    }
  }
}
