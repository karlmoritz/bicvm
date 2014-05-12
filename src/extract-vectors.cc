// File: extract-vectors.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 01-01-2013
// Last Update: Wed 07 May 2014 17:08:55 BST

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

#include "common/load_stanford.h"
#include "common/load_ccg.h"
#include "common/load_plain.h"

#include "common/senna.h"
#include "common/reindex_dict.h"
#include "common/utils.h"

#include "common/finetune_classifier.h"

// Training Regimes
#include "common/train_lbfgs.h"
#include "common/train_sgd.h"
#include "common/train_adagrad.h"

#include "common/train_update.h"
#include "common/finite_grad_check.h"

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
  cerr << "CCG-Based Deep Network: Copyright 2013 Karl Moritz Hermann" << endl;

  /***************************************************************************
   *                         Command line processing                         *
   ***************************************************************************/

  bpo::variables_map vm;

  // Command line processing
  bpo::options_description cmdline_specific("Command line specific options");
  cmdline_specific.add_options()
    ("help,h", "print help message")
    ("config,c", bpo::value<string>(),
     "config file specifying additional command line options")
    ;
  bpo::options_description generic("Allowed options");
  generic.add_options()
    ("type", bpo::value<string>()->default_value("ccaeb"),
     "type of model (ccaeb, mvrnn)")
    ("input", bpo::value<string>()->default_value(""),
     "vocabulary list for which vectors should be printed")
    ("model,m", bpo::value<string>(),
     "model to use")
    ("output", bpo::value<string>()->default_value(""),
     "vocabulary output file")
    ("dynamic-mode,d", bpo::value<int>()->default_value(1),
     "type of sentence representation: 0 (root+avg), 1 (root), 2 (avg), 3(concat-all), 4 (complicated)")
    ;
  bpo::options_description all_options;
  all_options.add(generic).add(cmdline_specific);

  store(parse_command_line(argc, argv, all_options), vm);
  if (vm.count("config") > 0) {
    ifstream config(vm["config"].as<string>().c_str());
    store(parse_config_file(config, all_options), vm);
  }
  notify(vm);

  if (vm.count("help")) {
    cout << all_options << "\n";
    return 1;
  }

  ModelData configA;

  RecursiveAutoencoderBase* raeptrA = nullptr;

  string type = vm["type"].as<string>();
  if (type == "additive") {
    raeptrA = new additive::RecursiveAutoencoder(configA);
  } else if (type == "flattree") {
    raeptrA = new flattree::RecursiveAutoencoder(configA);
  } else {
    cout << "Model (" << type << ") does not exist" << endl; return -1;
  }

  RecursiveAutoencoderBase& raeA = *raeptrA;

  if (vm.count("model"))
  {
    std::ifstream ifs(vm["model"].as<string>());
    boost::archive::text_iarchive ia(ifs);
    ia >> raeA;
  }

  /***************************************************************************
   *                   Print brief summary of model setup                    *
   ***************************************************************************/

  cerr << "################################" << endl;
  cerr << "# Config Summary" << endl;
  for (bpo::variables_map::iterator iter = vm.begin(); iter != vm.end(); ++iter)
  {
    cerr << "# " << iter->first << " = ";
    const ::std::type_info& type = iter->second.value().type() ;
    if ( type == typeid( ::std::string ) )
      cerr << iter->second.as<string>() << endl;
    if ( type == typeid( int ) )
      cerr << iter->second.as<int>() << endl;
    if ( type == typeid( Real ) )
      cerr << iter->second.as<Real>() << endl;
    if ( type == typeid( Real ) )
      cerr << iter->second.as<Real>() << endl;
    if ( type == typeid( bool ) )
      cerr << iter->second.as<bool>() << endl;
  }
  cerr << "# History" << endl << "# " << raeA.config.history << endl;
  cerr << "################################" << endl;

  /***************************************************************************
   *              Read in training data (positive and negative)              *
   ***************************************************************************/
  Dictionary dictA = raeA.getDictionary();

  std::ifstream infile(vm["input"].as<string>());
  std::string line;
  while (std::getline(infile, line)) {
    LabelID id = dictA.id(line);
    if (id != dictA.m_bad_label) {
      cout << line;
      for (int i = 0; i < raeA.config.word_representation_size; ++i) {
        cout << " " << raeA.D(id, i);
      }
      cout << endl;
    } else {
      cerr << "Not found in dict: " << line << endl;
    }
  }
}
