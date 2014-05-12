// File: ted-to-libsvm.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 01-01-2013
// Last Update: Mon 12 May 2014 18:15:32 BST

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
    ("mode", bpo::value<string>()->default_value("docprop"),
     "mode: docprop, average, baseline")
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

  // A bit of sanity checking options.
  // 1) If we run a baseline, we need (a) embeddings and (b) type additive.
  if (vm["mode"].as<string>() == "baseline") {
    assert(vm["embeddings"].as<int>() >= 0);
    assert(vm["type"].as<string>() == "additive");
    assert(vm.count("model") == 0);
  }
  // 2) If we don't run a baseline, we don't want any embeddings.
  if (vm["mode"].as<string>() == "docprop" || vm["mode"].as<string>() == "average")
    assert(vm["embeddings"].as<int>() < 0);

  assert(vm["mode"].as<string>() == "baseline"
         || vm["mode"].as<string>() == "docprop"
         || vm["mode"].as<string>() == "average");

  ModelData config;

  int embeddings = vm["embeddings"].as<int>();
  if (embeddings == 0)
    config.word_representation_size = 50;
  if (embeddings == 1)
    config.word_representation_size = 50;
  if (embeddings == 2 || embeddings == 3)
    config.word_representation_size = 40;
  if (embeddings > 3 && embeddings < 8)
    config.word_representation_size = 80;
  if (embeddings > 7)
    config.word_representation_size = 64;

  RecursiveAutoencoderBase* raeptrA = nullptr;
  RecursiveAutoencoderBase* docraeAptr = nullptr;

  string type = vm["type"].as<string>();
  if (type == "additive") {
    raeptrA = new additive::RecursiveAutoencoder(config);
    docraeAptr = new additive::RecursiveAutoencoder(config);
  } else if (type == "flattree") {
    raeptrA = new flattree::RecursiveAutoencoder(config);
    docraeAptr = new flattree::RecursiveAutoencoder(config);
  } else {
    cout << "Model (" << type << ") does not exist" << endl; return -1;
  }

  if ( vm["mode"].as<string>() == "average" ) {
    delete docraeAptr;
    docraeAptr = new additive::RecursiveAutoencoder(config);
  }

  RecursiveAutoencoderBase& raeA = *raeptrA;
  RecursiveAutoencoderBase& docraeA = *docraeAptr;

  /***************************************************************************
   *              Read in model and create raeA and docraeA                   *
   ***************************************************************************/

  if (vm.count("model")) {
    std::ifstream ifs(vm["model"].as<string>());
    boost::archive::text_iarchive ia(ifs);
    ia >> raeA;
    docraeA.config = raeA.config;
  }

  Model modelA, docmodA;
  Senna sennaA(raeA,embeddings);
  Senna sennadocA(docraeA,-1);
  string inputA = vm["input"].as<string>();
  bool add_to_dict = false;
  if (vm["mode"].as<string>() == "baseline")
    add_to_dict = true;
  load_doc::load_file(modelA.corpus, docmodA.corpus, inputA, add_to_dict, sennaA, sennadocA);
  docraeA.finalizeDictionary(true);

  if (add_to_dict) {
    raeA.finalizeDictionary(true);    // delete raeA theta and build new from scratch.
    raeA.D.setZero();                 // set all words to zero.
    sennaA.applyEmbeddings();         // use the embeddings for dict.
  }

  modelA.rae = reindex_dict(raeA,modelA.corpus);
  docmodA.rae = reindex_dict(docraeA,docmodA.corpus);
  modelA.finalize();
  docmodA.finalize();

  delete &raeA;
  delete &docraeA;

  /***************************************************************************
   *              Read in keytable (label_id <tab> label_name)               *
   ***************************************************************************/

  Dictionary keytable;
  std::vector<string> keylist;
  fs::path p(vm["keytable"].as<string>());
  assert(fs::exists(p) && fs::is_regular_file(p));
  std::string keyword, line;
  int id, check;
  fs::ifstream keyfile(p);
  while (getline(keyfile, line)) {
    std::stringstream ssline(line);
    ssline >> id;
    std::string word;
    keyword = "";
    while(ssline >> word){
      keyword = keyword + word;
    }
    check = keytable.id(keyword, true);
    keylist.push_back(keyword);
    assert(check == id);
  }

  /*
   * while (keyfile >> id >> word) {
   *   check = keytable.id(word, true);
   *   keylist.push_back(word);
   *   assert(check == id);
   * }
   */

  std::vector<string> document_keys;
  std::vector<std::vector<string>> document_keywords;
  fs::path sourceDir(inputA);
  fs::directory_iterator it(sourceDir), eod;
  BOOST_FOREACH(fs::path const &p, std::make_pair(it, eod)) {
    if(is_regular_file(p)) {
      fs::ifstream file(p);
      std::getline(file, line); // ignore first line
      std::getline(file, line); // this is the keywords line
      string doc_key;

      boost::regex rgx_one("<keywords>(.*)</keywords>");
      boost::smatch match;
      std::vector<string> keywords;
      if (boost::regex_search(line, match, rgx_one)) {
        line = match[1];
        // cout << "Matched: " << line << endl;

        boost::regex rgx("^,?([^,]+)");
        string::const_iterator begin = line.begin();
        string::const_iterator end = line.end();
        boost::match_results<string::const_iterator> what;
        while (regex_search(begin, end, what, rgx)) {
          std::string tmp = string(what[1].first, what[2].second);
          // boost::algorithm::trim(keyname);
          std::string keyname = regex_replace(tmp, boost::regex(" "), "");

          int key = keytable.id(keyname, false);
          if (key >= 0) {
            if (doc_key.length() > 0)  doc_key += ",";
            doc_key += boost::lexical_cast<std::string>(key);
            keywords.push_back(keyname);
          }
          begin = what[0].second;
        }
      } else {
        cout << "We have a problem with this file (no assert due to testing)" << endl;
      }
      document_keys.push_back(doc_key);
      document_keywords.push_back(keywords);
      // cout << "key: " << doc_key << endl;
    }
  }

  /***************************************************************************
   *                Create variables for forward propagation                 *
   ***************************************************************************/


  int asize = modelA.rae->getThetaSize();
  BackpropagatorBase* backprop = modelA.rae->getBackpropagator(modelA, asize);

  {
    // Forward propagate all sentences and store their resulting vector in
    // the document model dictionary.
    VectorReal x(modelA.rae->config.word_representation_size);
    for (int j = 0; j < modelA.corpus.size(); ++j) {
      backprop->forwardPropagate(j,&x);
      int sent_id = modelA.corpus[j].id;
      docmodA.rae->D.row(sent_id) = x;
    }
    delete backprop;
  }

  fs::path targetFile(vm["output"].as<string>());
  fs::ofstream outfile(targetFile);

  vector<shared_ptr<fs::ofstream>> key_outfiles;
  // std::vector<fs::ofstream> key_outfiles;
  for (auto key : keylist) {
    fs::path x(vm["output"].as<string>() + "-" + key);
    key_outfiles.push_back(make_shared<fs::ofstream>(x));
  }

  {
    // Forward propagate all documents and then print their resulting vectors
    // together with keywords and similar fun.
    int dsize = docmodA.rae->getThetaSize();
    BackpropagatorBase* docprop = docmodA.rae->getBackpropagator(docmodA, dsize);
    VectorReal x(modelA.rae->config.word_representation_size);
    for (int j = 0; j < docmodA.corpus.size(); ++j) {
      docprop->forwardPropagate(j,&x);
      outfile << document_keys[j];
        for (int k = 0; k < key_outfiles.size(); ++k) {
          if (std::find(document_keywords[j].begin(), document_keywords[j].end(), keylist[k]) != document_keywords[j].end())
            *key_outfiles[k] << "+1";
          else
            *key_outfiles[k] << "-1";
        }

      for (int i = 0; i < docmodA.rae->config.word_representation_size; ++i) {
        outfile << " " << i << ":" << x[i];
        for (int k = 0; k < key_outfiles.size(); ++k)
          *key_outfiles[k] << " " << i << ":" << x[i];
      }
      outfile << endl;
      for (int k = 0; k < key_outfiles.size(); ++k)
        *key_outfiles[k] << endl;
    }
    delete docprop;
  }
  outfile.close();
  for (int j = 0; j < key_outfiles.size(); ++j)
    key_outfiles[j]->close();
}
