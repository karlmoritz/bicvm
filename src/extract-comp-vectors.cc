// File: extract-comp-vectors.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 01-01-2013
// Last Update: Tue 14 Oct 2014 14:02:39 BST

// STL
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <queue>
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
#include <boost/iostreams/filter/gzip.hpp>
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

#include "common/load_stanford.h"
#include "common/load_ccg.h"
#include "common/load_plain.h"
#include "common/load_doc.h"

#include "common/senna.h"
#include "common/reindex_dict.h"
#include "common/utils.h"

#include "common/finetune_classifier.h"

// Training Regimes
#include "common/train_lbfgs.h"
#include "common/train_sgd.h"
#include "common/train_adagrad.h"
#include "common/finite_grad_check.h"

#include "common/trainer.h"
#include "common/openqa_trainer.h"
#include "common/general_trainer.h"

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
  cerr << "CCG-Based Deep Network: Copyright 2014 Karl Moritz Hermann" << endl;

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
     "type of model (ccaeb, mvrnn, ...)")
    ("input", bpo::value<string>()->default_value(""),
     "input directory containing *.ted files")
    ("model", bpo::value<string>(), "model")
    ("keytable", bpo::value<string>(), "keyword lookup table (identifier<tab>keyword)")
    ("output", bpo::value<string>()->default_value(""),
     "output file to which the libsvm training file will be written")
    // ("mode", bpo::value<string>()->default_value("docprop"),
     // "mode: docprop, average, baseline")
    ("embeddings", bpo::value<int>()->default_value(-1),
     "use embeddings for baseline dictionary (0=senna,1=turian,2=cldc-en,3=cldc-de)")
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
    if ( type == typeid( float ) )
      cerr << iter->second.as<float>() << endl;
    if ( type == typeid( double ) )
      cerr << iter->second.as<double>() << endl;
    if ( type == typeid( bool ) )
      cerr << iter->second.as<bool>() << endl;
  }
  cerr << "################################" << endl;

/*
 *   // A bit of sanity checking options.
 *   // 1) If we run a baseline, we need (a) embeddings and (b) type additive.
 *   if (vm["mode"].as<string>() == "baseline") {
 *     assert(vm["embeddings"].as<int>() >= 0);
 *     assert(vm["type"].as<string>() == "additive");
 *     assert(vm.count("model") == 0);
 *   }
 *   // 2) If we don't run a baseline, we don't want any embeddings.
 *   if (vm["mode"].as<string>() == "docprop" || vm["mode"].as<string>() == "average")
 *     assert(vm["embeddings"].as<int>() < 0);
 *
 *   assert(vm["mode"].as<string>() == "baseline"
 *          || vm["mode"].as<string>() == "docprop"
 *          || vm["mode"].as<string>() == "average");
 */

  ModelData config;

  /*
   * // Some code for loading baseline embeddings.
   * int embeddings = vm["embeddings"].as<int>();
   * if (embeddings == 0)
   *   config.word_representation_size = 50;
   * if (embeddings == 1)
   *   config.word_representation_size = 50;
   * if (embeddings == 2 || embeddings == 3)
   *   config.word_representation_size = 40;
   * if (embeddings > 3 && embeddings < 8)
   *   config.word_representation_size = 80;
   */

  RecursiveAutoencoderBase* raeptrA = nullptr;

  string type = vm["type"].as<string>();
  if (type == "additive") {
    raeptrA = new additive::RecursiveAutoencoder(config);
  } else if (type == "flattree") {
    raeptrA = new flattree::RecursiveAutoencoder(config);
  } else {
    cout << "Model (" << type << ") does not exist" << endl; return -1;
  }

  RecursiveAutoencoderBase& raeA = *raeptrA;
  DictionaryEmbeddings* deA = new DictionaryEmbeddings(config.word_representation_size);

  /***************************************************************************
   *              Read in model and create raeA and docraeA                   *
   ***************************************************************************/

  if (vm.count("model")) {
    std::ifstream ifs(vm["model"].as<string>());
    boost::archive::text_iarchive ia(ifs);
    ia >> raeA >> *deA;
  }

  Model modelA;
  Senna sennaA(*deA,-1); // embeddings);
  string inputA = vm["input"].as<string>();
  bool create_new_dict = false;
  /*
   * if (vm["mode"].as<string>() == "baseline")
   *   create_new_dict = true;
   */
  load_plain::load_file(modelA.corpus, inputA, 0, create_new_dict, sennaA);
  // docraeA.finalizeDictionary(true);

  /*
   * if (add_to_dict) {
   *   raeA.finalizeDictionary(true);    // delete raeA theta and build new from scratch.
   *   raeA.D.setZero();                 // set all words to zero.
   *   sennaA.applyEmbeddings();         // use the embeddings for dict.
   * }
   */

  if (create_new_dict) {
    raeA.createTheta(true); deA->createTheta(true); sennaA.applyEmbeddings(); }

  modelA.rae = &raeA;

  // Associate dictionaries with RAEs.
  modelA.rae->de_ = reindex_dict(*deA,modelA.corpus);
  delete deA;

  int ab_size = modelA.rae->getThetaSize()
              + modelA.rae->de_->getThetaSize();
  int offset = 0;
  Real* theta = new Real[ab_size]();
  modelA.rae->moveToAddress(theta + offset);
  offset += modelA.rae->getThetaSize();
  modelA.rae->de_->moveToAddress(theta + offset);

  modelA.finalize();

  /***************************************************************************
   *            Read in input file and create sentence strings               *
   ***************************************************************************/

  std::ifstream infile(vm["input"].as<string>());
  std::string line;
  std::vector<std::string> sentence_strings;
  while (std::getline(infile, line)) {
    trim(line);
    boost::replace_all(line, " ", "_");
    sentence_strings.push_back(line);
  }


  /***************************************************************************
   *                Create variables for forward propagation                 *
   ***************************************************************************/


  int asize = modelA.rae->getThetaPlusDictSize();
  BackpropagatorBase* backprop = modelA.rae->getBackpropagator(modelA, asize);

  fs::path targetFile(vm["output"].as<string>());
  fs::ofstream outfile(targetFile);

  {
    // Forward propagate all sentences and store their resulting vector in
    // the document model dictionary.
    VectorReal x(modelA.rae->config.word_representation_size);
    for (int j = 0; j < modelA.corpus.size(); ++j) {
      backprop->forwardPropagate(j,&x);
      outfile << sentence_strings[j];
      for (int i = 0; i < modelA.rae->config.word_representation_size; ++i) {
        outfile << " " << x[i];
      }
      outfile << endl;
    }
  }
  delete backprop;
  outfile.close();
}
