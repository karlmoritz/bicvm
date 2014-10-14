// File: doctrain.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 01-01-2013
// Last Update: Tue 14 Oct 2014 15:01:42 BST

// STL
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

// Boost
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
# if defined(RECENT_BOOST)
#include <boost/locale.hpp>
# endif

#include <Eigen/Core>

// Local
#include "common/shared_defs.h"
#include "common/dictionary.h"

#include "common/config.h"

#include "common/models.h"

#include "common/load_doc.h"

#include "common/senna.h"
#include "common/reindex_dict.h"

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

int main(int argc, char **argv)
{
  // Eigen::initParallel();
# if defined(RECENT_BOOST)
  boost::locale::generator gen;
  std::locale l = gen("de_DE.UTF-8");
  std::locale::global(l);
# endif

  cout << "BiCVM Distributed Representation Learner: Copyright 2013-2014 Karl Moritz Hermann" << endl;

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

    ("input1", bpo::value<string>()->default_value(""),
     "l1 corpus. sentence aligned to l2.")
    ("input2", bpo::value<string>()->default_value(""),
     "l2 corpus. sentence aligned to l1.")

    ("model1-in", bpo::value<string>(),
     "initial model 1")
    ("model2-in", bpo::value<string>(),
     "initial model 2")

    ("model1-out", bpo::value<string>()->default_value("modelA"),
     "base filename of model 1 output files")
    ("model2-out", bpo::value<string>()->default_value("modelB"),
     "base filename of model 2 output files")

    ("word-width", bpo::value<int>()->default_value(50),
     "width of word representation vectors.")
    ("iterations", bpo::value<int>()->default_value(-1),
     "(maximum) number of iterations (lbfgs default: 0 / sgd 250)")
    ("ftiterations", bpo::value<int>()->default_value(1000),
     "(maximum) number of finetune iterations")
    ("dump-frequency", bpo::value<int>()->default_value(10),
     "frequency at which to dump the model")
    ("num-sentences,n", bpo::value<int>()->default_value(0),
     "number of sentences to consider")
    ("method", bpo::value<string>()->default_value("lbfgs"),
     "training method (options: lbfgs,sgd,fgc,adagrad)")
    ("linesearch", bpo::value<string>()->default_value("armijo"),
     "LBFGS linesearch (morethuente, wolfe, armijo, strongwolfe)")

    ("embeddings", bpo::value<int>()->default_value(-1),
     "use embeddings to initialize dictionary (0=senna,1=turian,2=cldc)")

    ("batches", bpo::value<int>()->default_value(100),
     "number batches (adagrad minibatch)")
    ("ftcbatches", bpo::value<int>()->default_value(100),
     "number finetune batches (adagrad minibatch)")

    ("initI", bpo::value<bool>()->default_value(true),
     "initialize weight matrices to partial identity?")

    ("updateD1", bpo::value<bool>()->default_value(true), "learn D weights?")
    ("updateF1", bpo::value<bool>()->default_value(true), "learn F weights?")
    //("updateA1", bpo::value<bool>()->default_value(true), "learn A weights?")
    ("updateWd1", bpo::value<bool>()->default_value(true), "learn Wd weights?")
    //("updateWf1", bpo::value<bool>()->default_value(true), "learn Wf weights?")
    ("updateWl1", bpo::value<bool>()->default_value(true), "learn Wl weights?")

    ("updateD2", bpo::value<bool>()->default_value(true), "learn D weights for B?")
    ("updateF2", bpo::value<bool>()->default_value(true), "learn F weights for B?")
    //("updateA2", bpo::value<bool>()->default_value(true), "learn A weights for B?")
    ("updateWd2", bpo::value<bool>()->default_value(true), "learn Wd weights for B?")
    //("updateWf2", bpo::value<bool>()->default_value(true), "learn Wf weights for B")
    ("updateWl2", bpo::value<bool>()->default_value(true), "learn Wl weights for B?")

    ("calc_rae_error1", bpo::value<bool>()->default_value(false),
     "consider the reconstruction error?")
    ("calc_lbl_error1", bpo::value<bool>()->default_value(false),
     "consider the label error?")
    ("calc_bi_error1", bpo::value<bool>()->default_value(false),
     "consider the bi error (matching root)?")
    ("calc_thr_error1", bpo::value<bool>()->default_value(false),
     "consider the throughprop error?")
    ("calc_uae_error1", bpo::value<bool>()->default_value(false),
     "consider the unfolding error?")

    ("calc_rae_error2", bpo::value<bool>()->default_value(false),
     "consider the reconstruction error?")
    ("calc_lbl_error2", bpo::value<bool>()->default_value(false),
     "consider the label error?")
    ("calc_bi_error2", bpo::value<bool>()->default_value(false),
     "consider the bi error (matching root)?")
    ("calc_thr_error2", bpo::value<bool>()->default_value(false),
     "consider the throughprop error?")
    ("calc_uae_error2", bpo::value<bool>()->default_value(false),
     "consider the unfolding error?")

    /*
     *("fonly", bpo::value<bool>()->default_value(false),
     * "only finetune model (no RAE training)")
     *("f-in", bpo::value<string>(),
     * "finetune model in")
     *("f-out", bpo::value<string>()->default_value("fmodel"),
     * "finetune model out")
     */

    ("lambdaD", bpo::value<Real>()->default_value(1), "L2 Regularization for Embeddings")
    //("lambdaF", bpo::value<Real>()->default_value(0.000001), "Regularization for Embeddings")
    //("lambdaA", bpo::value<Real>()->default_value(0.0000), "Regularization for Embeddings")
    ("lambdaWd", bpo::value<Real>()->default_value(1), "L2 Regularization for Tree Matrices")
    ("lambdaBd", bpo::value<Real>()->default_value(1), "L2 Regularization for Tree Biases")
    //("lambdaWf", bpo::value<Real>()->default_value(0.007), "Regularization for Tree Matrices")
    ("lambdaWl", bpo::value<Real>()->default_value(1), "L2 Regularization for Label Matrices")
    ("lambdaBl", bpo::value<Real>()->default_value(1), "L2 Regularization for Label Biases")
    ("l1", bpo::value<Real>()->default_value(0), "L1 Regularization")

    ("alpha", bpo::value<Real>()->default_value(0.2),
     "autoencoder error vs label error")
    ("gamma", bpo::value<Real>()->default_value(0.1),
     "noisy error as percentage of normal bi-error")
    ("epsilon", bpo::value<Real>()->default_value(0.000001),
     "convergence parameter for LBFGS")
    ("eta", bpo::value<Real>()->default_value(0.2),
     "(initial) eta for SGD")
    ("ftceta", bpo::value<Real>()->default_value(0.01),
     "(initial) eta for finetune AdaGrad")

    ("norm", bpo::value<int>()->default_value(0),
     "normalization type (see train_update.cc)")
    ("dynamic-mode,d", bpo::value<int>()->default_value(0),
     "type of sentence representation: 0 (root+avg), 1 (root), 2 (avg), 3(concat-all), 4 (complicated)")

    ("noise", bpo::value<int>()->default_value(2),
     "number of noise samples per positive training example")
    ("hinge_loss_margin", bpo::value<Real>()->default_value(1.0), "Hinge loss margin")
    ;
  bpo::options_description all_options;
  all_options.add(generic).add(cmdline_specific);

  bpo::parsed_options opts = parse_command_line(argc, argv, all_options);
  store(opts, vm);
  /*
   * if (vm.count("config") > 0) {
   *   ifstream config(vm["config"].as<string>().c_str());
   *   store(parse_config_file(config, all_options), vm);
   * }
   */
  notify(vm);

  if (vm.count("help")) {
    cout << all_options << "\n";
    return 1;
  }


  ModelData config;

  config.word_representation_size = vm["word-width"].as<int>();
  config.num_sentences = vm["num-sentences"].as<int>();

  //config.model_out = vm["model1-out"].as<string>();
  config.dump_freq = vm["dump-frequency"].as<int>();
  config.init_to_I = vm["initI"].as<bool>();

  config.tree = TREE_PLAIN;

  if (vm["method"].as<string>() == "lbfgs")
    config.training_method = 0;
  else if (vm["method"].as<string>() == "sgd")
    config.training_method = 1;
  else if (vm["method"].as<string>() == "fgc")
    config.training_method = 2;
  else if (vm["method"].as<string>() == "adagrad")
    config.training_method = 3;
  else if (vm["method"].as<string>() == "lbfgs-mb")
    config.training_method = 4;
  else
    config.training_method = 0;

  config.hinge_loss_margin = vm["hinge_loss_margin"].as<Real>();

  Bools bools1;
  bools1.D   = vm["updateD1"].as<bool>();
  bools1.U   = vm["updateF1"].as<bool>();
  bools1.V   = vm["updateF1"].as<bool>();
  bools1.W   = vm["updateF1"].as<bool>();
  bools1.Wd  = vm["updateWd1"].as<bool>();
  bools1.Wdr = vm["updateWd1"].as<bool>();
  bools1.Bd  = vm["updateWd1"].as<bool>();
  bools1.Bdr = vm["updateWd1"].as<bool>();
  bools1.Wl  = vm["updateWl1"].as<bool>();
  bools1.Bl  = vm["updateWl1"].as<bool>();

  Bools bools2;
  bools2.D   = vm["updateD2"].as<bool>();
  bools2.U   = vm["updateF2"].as<bool>();
  bools2.V   = vm["updateF2"].as<bool>();
  bools2.W   = vm["updateF2"].as<bool>();
  bools2.Wd  = vm["updateWd2"].as<bool>();
  bools2.Wdr = vm["updateWd2"].as<bool>();
  bools2.Bd  = vm["updateWd2"].as<bool>();
  bools2.Bdr = vm["updateWd2"].as<bool>();
  bools2.Wl  = vm["updateWl2"].as<bool>();
  bools2.Bl  = vm["updateWl2"].as<bool>();

  // Split the config into two separate configurations for A and B
  ModelData configA(config);
  ModelData configB(config);

  configA.model_out = vm["model1-out"].as<string>();
  configB.model_out = vm["model2-out"].as<string>();

  configA.calc_rae     = vm["calc_rae_error1"].as<bool>();
  configA.calc_lbl     = vm["calc_lbl_error1"].as<bool>();
  configA.calc_bi      = vm["calc_bi_error1"].as<bool>();
  configA.calc_through = vm["calc_thr_error1"].as<bool>();
  configA.calc_uae     = vm["calc_uae_error1"].as<bool>();

  configB.calc_rae     = vm["calc_rae_error2"].as<bool>();
  configB.calc_lbl     = vm["calc_lbl_error2"].as<bool>();
  configB.calc_bi      = vm["calc_bi_error2"].as<bool>();
  configB.calc_through = vm["calc_thr_error2"].as<bool>();
  configB.calc_uae     = vm["calc_uae_error2"].as<bool>();

  bool fonly      = false; //vm["fonly"].as<bool>();

  Real eta       = vm["eta"].as<Real>();
  Real ftceta    = vm["ftceta"].as<Real>();
  Real alpha     = vm["alpha"].as<Real>();
  Real gamma     = vm["gamma"].as<Real>();
  Real epsilon   = vm["epsilon"].as<Real>();

  Lambdas lambdas;
  lambdas.D   = vm["lambdaD"].as<Real>();
  lambdas.Wd  = vm["lambdaWd"].as<Real>();
  lambdas.Wdr = vm["lambdaWd"].as<Real>();
  lambdas.Bd  = vm["lambdaBd"].as<Real>();
  lambdas.Bdr = vm["lambdaBd"].as<Real>();
  lambdas.Wl  = vm["lambdaWl"].as<Real>();
  lambdas.Bl  = vm["lambdaBl"].as<Real>();

  lambdas.alpha_rae = 1.0;
  lambdas.alpha_lbl = 1.0; //(1.0 - alpha);

  Real l1 = vm["l1"].as<Real>();

  int dmode         = vm["dynamic-mode"].as<int>();

  int batches       = vm["batches"].as<int>();
  int ftcbatches    = vm["ftcbatches"].as<int>();


  LineSearchType linesearch =s2line_map[vm["linesearch"].as<string>()];

  int iterations = vm["iterations"].as<int>();
  int ftiterations = vm["ftiterations"].as<int>();
  if (iterations == -1)
  {
    if (config.training_method == 1)
      iterations = 250;
    else if (config.training_method == 3)
      iterations = 10;
    else
      iterations = 0;
  }

  //config.num_weight_types = num_ccg_rules + num_cat_rules + 1;

/***************************************************************************
 *          Create RAE Instance and load data if model is present          *
 ***************************************************************************/

  bool create_new_dict_A = true;
  bool create_new_dict_B = true;

  RecursiveAutoencoderBase* raeptrA = nullptr;
  RecursiveAutoencoderBase* raeptrB = nullptr;
  RecursiveAutoencoderBase* docraeptrA = nullptr;
  RecursiveAutoencoderBase* docraeptrB = nullptr;

  string type = vm["type"].as<string>();
  if (type == "additive") {
    raeptrA = new additive::RecursiveAutoencoder(configA);
    raeptrB = new additive::RecursiveAutoencoder(configB);
    docraeptrA = new additive::RecursiveAutoencoder(configA);
    docraeptrB = new additive::RecursiveAutoencoder(configB);
  } else if (type == "flattree") {
    raeptrA = new flattree::RecursiveAutoencoder(configA);
    raeptrB = new flattree::RecursiveAutoencoder(configB);
    docraeptrA = new flattree::RecursiveAutoencoder(configA);
    docraeptrB = new flattree::RecursiveAutoencoder(configB);
  } else {
    cout << "Model (" << type << ") does not exist" << endl; return -1;
  }

  RecursiveAutoencoderBase& raeA = *raeptrA;
  RecursiveAutoencoderBase& raeB = *raeptrB;
  RecursiveAutoencoderBase& docraeA = *docraeptrA;
  RecursiveAutoencoderBase& docraeB = *docraeptrB;
  DictionaryEmbeddings* deA = new DictionaryEmbeddings(config.word_representation_size);
  DictionaryEmbeddings* deB = new DictionaryEmbeddings(config.word_representation_size);
  DictionaryEmbeddings* docdeA = new DictionaryEmbeddings(config.word_representation_size);
  DictionaryEmbeddings* docdeB = new DictionaryEmbeddings(config.word_representation_size);

  if (vm.count("model1-in"))
  {
    std::ifstream ifs(vm["model1-in"].as<string>());
    boost::archive::text_iarchive ia(ifs);
    ia >> raeA >> *deA;
    create_new_dict_A = false;
  }
  if (vm.count("model2-in"))
  {
    std::ifstream ifs(vm["model2-in"].as<string>());
    boost::archive::text_iarchive ia(ifs);
    ia >> raeB >> *deB;
    create_new_dict_B = false;
  }

  // Update the "history" maker in the config file
  std::stringstream ss;
  // TODO separate history streams A and B
  ss << raeA.config.history << " | " << vm["method"].as<string>();
  if (config.training_method != 1)
    ss << "(" << vm["linesearch"].as<string>() << ")";
  ss << " it:" << iterations; // << " wcat/only:" << wcat << " " << wcatonly;
  ss << " lambdas: "  << "/" << lambdas.Wl << "/";
  raeA.config.history = ss.str();
  raeB.config.history = ss.str();

  /***************************************************************************
   *                   Print brief summary of model setup                    *
   ***************************************************************************/

  cerr << "################################" << endl;
  cerr << "# Config Summary" << endl;
  for (bpo::variables_map::iterator iter = vm.begin(); iter != vm.end(); ++iter) {
    cerr << "# " << iter->first << " = ";
    const ::std::type_info& type = iter->second.value().type() ;
    if ( type == typeid( ::std::string ) )
      cerr << iter->second.as<string>() << endl;
    if ( type == typeid( int ) )
      cerr << iter->second.as<int>() << endl;
    if ( type == typeid( double ) )
      cerr << iter->second.as<double>() << endl;
    if ( type == typeid( float ) )
      cerr << iter->second.as<float>() << endl;
    if ( type == typeid( bool ) )
      cerr << iter->second.as<bool>() << endl;
  }
  cerr << "# History" << endl << "# " << raeA.config.history << endl;
  cerr << "################################" << endl;

  /***************************************************************************
   *              Read in training data (positive and negative)              *
   ***************************************************************************/

  Model modelA, docmodA;
  Model modelB, docmodB;

  string inputA = vm["input1"].as<string>();
  string inputB = vm["input2"].as<string>();

  // No embeddings for B unless we use Klementiev/Titov 2012 embeddings de/en
  int embeddings_typeA = vm["embeddings"].as<int>();
  int embeddings_typeB = -1;
  if (embeddings_typeA == 2) embeddings_typeB = 3;

  {
    Senna sennaA(*deA,embeddings_typeA);
    Senna sennaB(*deB,embeddings_typeB);
    Senna sennadocA(*docdeA,-1);
    Senna sennadocB(*docdeB,-1);
    //        where,       filename,       cv, use_cv, test?, label, add?,       senna?
    if (!inputA.empty())
      load_doc::load_file(modelA.corpus, docmodA.corpus, inputA, create_new_dict_A, sennaA, sennadocA);
    if (!inputB.empty())
      load_doc::load_file(modelB.corpus, docmodB.corpus, inputB, create_new_dict_B, sennaB, sennadocB);

    cout << "L1 Size " << modelA.corpus.size() << endl;
    cout << "L2 Size " << modelB.corpus.size() << endl;

    // (Re)create dictionary space and add senna embeddings.
    if (create_new_dict_A) {
      raeA.createTheta(true); deA->createTheta(true); sennaA.applyEmbeddings(); }
    if (create_new_dict_B) {
      raeB.createTheta(true); deB->createTheta(true); sennaB.applyEmbeddings(); }

    docraeA.createTheta(true); docdeA->createTheta(true);
    docraeB.createTheta(true); docdeB->createTheta(true);
  }

  /***************************************************************************
   *           Setup model and update dictionary if created above            *
   ***************************************************************************/

  modelA.rae = &raeA;
  modelB.rae = &raeB;
  docmodA.rae = &docraeA;
  docmodB.rae = &docraeB;
  // Hacky: alphas directly into model.
  modelA.rae->alpha_rae = lambdas.alpha_rae;
  modelA.rae->alpha_lbl = lambdas.alpha_lbl;
  modelB.rae->alpha_rae = lambdas.alpha_rae;
  modelB.rae->alpha_lbl = lambdas.alpha_lbl;
  docmodA.rae->alpha_rae = lambdas.alpha_rae;
  docmodA.rae->alpha_lbl = lambdas.alpha_lbl;
  docmodB.rae->alpha_rae = lambdas.alpha_rae;
  docmodB.rae->alpha_lbl = lambdas.alpha_lbl;

  // Associate dictionaries with RAEs.
  modelA.rae->de_ = reindex_dict(*deA,modelA.corpus);
  modelB.rae->de_ = reindex_dict(*deB,modelB.corpus);
  docmodA.rae->de_ = reindex_dict(*docdeA,docmodA.corpus);
  docmodB.rae->de_ = reindex_dict(*docdeB,docmodB.corpus);
  delete deA;
  delete deB;
  delete docdeA;
  delete docdeB;

  int ab_size = modelA.rae->getThetaSize()
              + modelB.rae->getThetaSize()
              + modelA.rae->de_->getThetaSize()
              + modelB.rae->de_->getThetaSize();
  int offset = 0;
  Real* theta = new Real[ab_size]();
  modelA.rae->moveToAddress(theta + offset);
  offset += modelA.rae->getThetaSize();
  modelB.rae->moveToAddress(theta + offset);
  offset += modelB.rae->getThetaSize();
  modelA.rae->de_->moveToAddress(theta + offset);
  offset += modelA.rae->de_->getThetaSize();
  modelB.rae->de_->moveToAddress(theta + offset);

  // Sets some model parameters such as maximum sentence and node length to
  // speed up memory access and parallelisation later on.
  modelA.finalize();
  modelB.finalize();

  ab_size = docmodA.rae->getThetaSize()
          + docmodB.rae->getThetaSize()
          + docmodA.rae->de_->getThetaSize()
          + docmodB.rae->de_->getThetaSize();
  offset = 0;
  Real* theta2 = new Real[ab_size]();
  docmodA.rae->moveToAddress(theta2 + offset);
  offset += docmodA.rae->getThetaSize();
  docmodB.rae->moveToAddress(theta2 + offset);
  offset += docmodB.rae->getThetaSize();
  docmodA.rae->de_->moveToAddress(theta2 + offset);
  offset += docmodA.rae->de_->getThetaSize();
  docmodB.rae->de_->moveToAddress(theta2 + offset);

  docmodA.finalize();
  docmodB.finalize();

  modelA.bools = bools1;
  modelA.alpha = alpha;
  modelA.gamma = gamma;

  modelA.from = 0;
  modelA.to = modelA.corpus.size();
  modelA.normalization_type = vm["norm"].as<int>();
  modelA.num_noise_samples = vm["noise"].as<int>();

  modelB.bools = bools2;
  modelB.alpha = alpha;
  modelB.gamma = gamma;

  modelB.from = 0;
  modelB.to = modelB.corpus.size();
  modelB.normalization_type = vm["norm"].as<int>();
  modelB.num_noise_samples = vm["noise"].as<int>();

  docmodA.from = 0;
  docmodA.to = docmodA.corpus.size();
  docmodB.from = 0;
  docmodB.to = docmodA.corpus.size();

  docmodA.bools = bools1;
  docmodA.alpha = alpha;
  docmodA.gamma = gamma;

  docmodB.bools = bools2;
  docmodB.alpha = alpha;
  docmodB.gamma = gamma;

  if (config.num_sentences != 0) {
    cout << "Reset to small" << endl;
    modelA.to = min(int(modelA.corpus.size()),config.num_sentences);
    modelB.to = min(int(modelB.corpus.size()),config.num_sentences);
  }

  cout << "Dict size: " << modelA.rae->getDictSize() << " and " << modelB.rae->getDictSize() << endl;

  modelA.trainer = new GeneralTrainer();

  /***************************************************************************
   *                              BFGS training                              *
   ***************************************************************************/

  modelA.b = &modelB;
  modelA.docmod = &docmodA;
  modelB.docmod = &docmodB;

  if (config.training_method == 0)
  {
    cout << "Training with LBFGS" << endl;
    train_lbfgs(modelA,linesearch,iterations,epsilon,lambdas);
  }
  else if (config.training_method == 1)
  {
    cout << "Training with SGD" << endl;
    train_sgd(modelA,iterations,eta,lambdas);
  }
  else if (config.training_method == 2)
  {
    cout << "Finite Gradient Check" << endl;
    finite_bigrad_check(modelA,lambdas);
  }
  else if (config.training_method == 3)
  {
    cout << "Training with AdaGrad" << endl;
    //Model tmodel(modelA.rae,testCorpus);
    train_adagrad(modelA,iterations,eta,nullptr,batches,lambdas,l1);
  }
  if (config.training_method == 4)
  {
    cout << "Training with LBFGS (minibatch)" << endl;
    train_lbfgs_minibatch(modelA,linesearch,iterations,epsilon,batches,lambdas);
  }

  /***************************************************************************
   *                 Storing model to file (default: model)                  *
   ***************************************************************************/

  {
    std::ofstream ofs(vm["model1-out"].as<string>());
    boost::archive::text_oarchive oa(ofs);
    oa << *(modelA.rae) << *(modelA.rae->de_);
  }

  {
    std::ofstream ofs(vm["model2-out"].as<string>());
    boost::archive::text_oarchive oa(ofs);
    oa << *(modelB.rae) << *(modelB.rae->de_);
  }
}
