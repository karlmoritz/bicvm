// File: qq-extract-relent.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 01-01-2013
// Last Update: Tue 14 Oct 2014 15:08:59 BST

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
#include "common/load_relent.h"
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
  cerr << "QQ relation and entity representation extractor: Copyright 2014 Karl Moritz Hermann" << endl;

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
     "prefix for output files <pre>.ent and <pre>.rel")
    ("embeddings", bpo::value<int>()->default_value(-1),
     "use embeddings for baseline dictionary (0=senna,1=turian,2=cldc-en,3=cldc-de)")
    ("format", bpo::value<int>()->default_value(0),
     "output format: 0=space separate, 1=tab separator")
    ("header", bpo::value<bool>()->default_value(false),
     "include header row")
    ("external-names", bpo::value<bool>()->default_value(false),
     "include names as first column OR output into separate files")
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

  Model modelA;
  int embeddings = vm["embeddings"].as<int>();
  string inputA = vm["input"].as<string>();
  bool create_new_dict = false;

  Senna sennaA(*deA,embeddings);

  assert(!inputA.empty());
  // Load with relent (treat as set of strings) or plain (treat as unique type)
  if ( vm["eas"].as<bool>() )
    load_relent::load_file(modelA.corpus, inputA, create_new_dict, sennaA);
  else
    load_plain::load_file(modelA.corpus, inputA, 0, create_new_dict, sennaA);

  modelA.rae = &raeA;
  modelA.rae->de_ = deA;

  /***************************************************************************
   *            Read in input file and create sentence strings               *
   ***************************************************************************/

  std::ifstream infile(vm["input"].as<string>());
  std::string line;
  std::vector<std::string> sentence_strings;
  std::vector<std::string> types;
  while (std::getline(infile, line)) {
    trim(line);
    type = line.substr(line.length()-1,1);
    line = line.substr(line.find(" ")+1);
    sentence_strings.push_back(line);
    types.push_back(type);
  }

  /***************************************************************************
   *    Propagate inputs and store in appropriate output files               *
   ***************************************************************************/

  int asize = modelA.rae->getThetaPlusDictSize();
  BackpropagatorBase* backprop = modelA.rae->getBackpropagator(modelA, asize);

  int word_width = modelA.rae->config.word_representation_size;
  fs::path rel_file_path(vm["prefix"].as<string>() + ".rel");
  fs::path ent_file_path(vm["prefix"].as<string>() + ".ent");
  fs::ofstream rel_file(rel_file_path);
  fs::ofstream ent_file(ent_file_path);
  fs::path relname_file_path(vm["prefix"].as<string>() + ".relname");
  fs::path entname_file_path(vm["prefix"].as<string>() + ".entname");
  fs::ofstream relname_file(relname_file_path);
  fs::ofstream entname_file(entname_file_path);

  string separator = " ";
  if (vm["format"].as<int>() == 1)
    separator = "\t";

  {
    // Forward propagate all sentences and store their resulting vector in
    // the document model dictionary.
    if (vm["header"].as<bool>()) {
      if(!vm["external-names"].as<bool>()) {
        rel_file << "Name" << "\t";
        ent_file << "Name" << "\t";
      } else {
        relname_file << "Name" << endl;
        entname_file << "Name" << endl;
      }
      rel_file << 0;
      ent_file << 0;
      for (int j = 1; j < word_width; ++j) {
        rel_file << "\t" << j;
        ent_file << "\t" << j;
      }
      rel_file << endl;
      ent_file << endl;
    }
    VectorReal x(word_width);
    for (int j = 0; j < modelA.corpus.size(); ++j) {
      // CHECK IF WORD EXISTS IN DICTIONARY FIRST! ( in !eas case)
      if (vm["eas"].as<bool>() or (deA->dict_.id(sentence_strings[j]) != deA->dict_.m_bad_label)) {
        backprop->forwardPropagate(j,&x);
        if (types[j] == "e" or types[j] == "1" or types[j] == "2") {
          if(!vm["external-names"].as<bool>()) {
            ent_file << sentence_strings[j] << separator;
          } else {
            entname_file << sentence_strings[j] << endl;
          }
          ent_file << x[0];
          for (int i = 1; i < word_width; ++i) { ent_file << separator << x[i]; }
          ent_file << endl;
        }
        if (types[j] == "r") {
          if(!vm["external-names"].as<bool>()) {
            rel_file << sentence_strings[j] << separator;
          } else {
            relname_file << sentence_strings[j] << endl;
          }
          rel_file << x[0];
          for (int i = 1; i < word_width; ++i) { rel_file << separator << x[i]; }
          rel_file << endl;
        }
      }
    }
  }
  delete backprop;
  rel_file.close();
  ent_file.close();
  relname_file.close();
  entname_file.close();
}
