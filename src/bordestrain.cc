// File: bordestrain.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 01-01-2013
// Last Update: Thu 29 May 2014 13:48:32 BST

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

#include "common/load_qq.h"

#include "common/senna.h"
#include "common/reindex_dict.h"


// Training Regimes
#include "common/train_lbfgs.h"
#include "common/train_sgd.h"
#include "common/train_adagrad.h"
#include "common/finite_grad_check.h"

#include "common/trainer.h"
#include "common/openqa_bordes_trainer.h"

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

  cout << "Siamese-style Network for Question-Query Embedding Learning" << endl
    << "Copyright 2014 Karl Moritz Hermann" << endl;

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
    ("type", bpo::value<string>()->default_value("additive"),
     "type of model (additive, flattree, ...)")
    ("input", bpo::value<string>()->default_value(""),
     "question query corpus (tab separated question query).")
    ("input2", bpo::value<string>()->default_value(""),
     "paraphrase corpus (tab separated sentences).")

    ("model-in", bpo::value<string>(),
     "initial model")
    ("model-out", bpo::value<string>()->default_value("modelA"),
     "base filename of model output files")

    ("word-width", bpo::value<int>()->default_value(50),
     "width of word representation vectors.")
    ("iterations", bpo::value<int>()->default_value(-1),
     "(maximum) number of iterations (lbfgs default: 0 / sgd 250)")
    ("dump-frequency", bpo::value<int>()->default_value(10),
     "frequency at which to dump the model")
    // ("num-sentences,n", bpo::value<int>()->default_value(0),
     // "number of sentences to consider")
    ("method", bpo::value<string>()->default_value("adagrad"),
     "training method (options: lbfgs,sgd,fgc,adagrad)")
    // ("linesearch", bpo::value<string>()->default_value("armijo"),
     // "LBFGS linesearch (morethuente, wolfe, armijo, strongwolfe)")

    ("embeddings", bpo::value<int>()->default_value(-1),
     "use embeddings to initialize dictionary (0=senna,1=turian,2=cldc,...)")

    ("batches", bpo::value<int>()->default_value(100),
     "number batches (adagrad minibatch)")

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
    ("noise", bpo::value<int>()->default_value(2),
     "number of noise samples per positive training example")
    ("hinge_loss_margin", bpo::value<Real>()->default_value(1.0), "Hinge loss margin")
    ;
  bpo::options_description all_options;
  all_options.add(generic).add(cmdline_specific);

  bpo::parsed_options opts = parse_command_line(argc, argv, all_options);
  store(opts, vm);
  notify(vm);

  if (vm.count("help")) {
    cout << all_options << "\n";
    return 1;
  }


  ModelData config;

  config.word_representation_size = vm["word-width"].as<int>();
  config.num_sentences = 0; //vm["num-sentences"].as<int>();
  config.dump_freq = vm["dump-frequency"].as<int>();
  config.init_to_I = vm["initI"].as<bool>();
  config.hinge_loss_margin = vm["hinge_loss_margin"].as<Real>();

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

  configA.model_out = vm["model-out"].as<string>();

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

  bool throughprop = (configA.calc_through or configB.calc_through) ? true : false;

  bool fonly      = false; //vm["fonly"].as<bool>();

  Real eta       = vm["eta"].as<Real>();
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

  int batches       = vm["batches"].as<int>();

  // LineSearchType linesearch =s2line_map[vm["linesearch"].as<string>()];

  int iterations = vm["iterations"].as<int>();
  if (iterations == -1)
  {
    if (config.training_method == 1)
      iterations = 250;
    else if (config.training_method == 3)
      iterations = 10;
    else
      iterations = 0;
  }

/***************************************************************************
 *          Create RAE Instance and load data if model is present          *
 ***************************************************************************/

  bool create_new_dict = true;

  RecursiveAutoencoderBase* raeptrA = nullptr;
  RecursiveAutoencoderBase* raeptrB = nullptr;
  RecursiveAutoencoderBase* raeptrPPA = nullptr;
  RecursiveAutoencoderBase* raeptrPPB = nullptr;

  string type = vm["type"].as<string>();
  if (type == "additive") {
    raeptrA = new additive::RecursiveAutoencoder(configA);
    raeptrB = new additive::RecursiveAutoencoder(configB);
    raeptrPPA = new additive::RecursiveAutoencoder(configA);
    raeptrPPB = new additive::RecursiveAutoencoder(configB);
  } else if (type == "flattree") {
    raeptrA = new flattree::RecursiveAutoencoder(configA);
    raeptrB = new flattree::RecursiveAutoencoder(configB);
    raeptrPPA = new flattree::RecursiveAutoencoder(configA);
    raeptrPPB = new flattree::RecursiveAutoencoder(configB);
  } else if (type == "additive_avg") {
    raeptrA = new additive_avg::RecursiveAutoencoder(configA);
    raeptrB = new additive_avg::RecursiveAutoencoder(configB);
    raeptrPPA = new additive_avg::RecursiveAutoencoder(configA);
    raeptrPPB = new additive_avg::RecursiveAutoencoder(configB);
  } else {
    cout << "Model (" << type << ") does not exist" << endl; return -1;
  }

  RecursiveAutoencoderBase& raeA = *raeptrA;
  RecursiveAutoencoderBase& raeB = *raeptrB;
  RecursiveAutoencoderBase& raePPA = *raeptrPPA;
  RecursiveAutoencoderBase& raePPB = *raeptrPPB;
  DictionaryEmbeddings* deA = new DictionaryEmbeddings(config.word_representation_size);

  if (vm.count("model-in"))
  {
    std::ifstream ifs(vm["model-in"].as<string>());
    boost::archive::text_iarchive ia(ifs);
    ia >> raeA >> *deA;
    create_new_dict = false;
  }

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
  cerr << "################################" << endl;

  /***************************************************************************
   *              Read in training data (positive and negative)              *
   ***************************************************************************/

  Model modelA, modelB, modelPPA, modelPPB;

  string input = vm["input"].as<string>();
  string input2 = vm["input2"].as<string>();

  int embeddings_typeA = vm["embeddings"].as<int>();
  int embeddings_typeB = -1;
  if (embeddings_typeA == 2) embeddings_typeB = 3;

  {
    Senna sennaA(*deA,embeddings_typeA);
    // Senna sennaB(*deB,embeddings_typeB);
    assert(!input.empty());
    load_qq::load_file(modelA.corpus, modelB.corpus, input, create_new_dict,
                       create_new_dict, sennaA, sennaA);
    load_qq::load_file(modelPPA.corpus, modelPPB.corpus, input2, create_new_dict,
                       create_new_dict, sennaA, sennaA);

    cout << "L1 Size " << modelA.corpus.size() << endl;
    cout << "L2 Size " << modelB.corpus.size() << endl;
    cout << "L3 Size " << modelPPA.corpus.size() << endl;
    cout << "L4 Size " << modelPPB.corpus.size() << endl;

    // (Re)create dictionary space and add senna embeddings.
    if (create_new_dict) {
      raeA.createTheta(true); deA->createTheta(true); sennaA.applyEmbeddings();
      raeB.createTheta(true);
      raePPA.createTheta(true);
      raePPB.createTheta(true);
    }
  }

  /***************************************************************************
   *            Setup model and update dictionary if create above            *
   ***************************************************************************/

  modelA.rae = &raeA;
  modelB.rae = &raeB;
  modelPPA.rae = &raePPA;
  modelPPB.rae = &raePPB;
  // Hacky: alphas directly into model.
  modelA.rae->alpha_rae = lambdas.alpha_rae;
  modelA.rae->alpha_lbl = lambdas.alpha_lbl;
  modelB.rae->alpha_rae = lambdas.alpha_rae;
  modelB.rae->alpha_lbl = lambdas.alpha_lbl;
  modelPPA.rae->alpha_rae = lambdas.alpha_rae;
  modelPPA.rae->alpha_lbl = lambdas.alpha_lbl;
  modelPPB.rae->alpha_rae = lambdas.alpha_rae;
  modelPPB.rae->alpha_lbl = lambdas.alpha_lbl;

  // Associate dictionaries with RAEs.
  modelA.rae->de_ = deA; //reindex_dict(*deA,modelA.corpus);
  modelB.rae->de_ = deA; //modelA.rae->de_; //reindex_dict(*deB,modelB.corpus);
  modelPPA.rae->de_ = deA; //modelA.rae->de_; //reindex_dict(*deB,modelB.corpus);
  modelPPB.rae->de_ = deA; //modelA.rae->de_; //reindex_dict(*deB,modelB.corpus);

  int ab_size = modelA.rae->getThetaSize()
              + modelB.rae->getThetaSize()
              + modelPPA.rae->getThetaSize()
              + modelPPB.rae->getThetaSize()
              + modelA.rae->de_->getThetaSize();
  int offset = 0;
  Real* theta = new Real[ab_size]();
  modelA.rae->moveToAddress(theta + offset);
  offset += modelA.rae->getThetaSize();
  modelB.rae->moveToAddress(theta + offset);
  offset += modelB.rae->getThetaSize();
  modelPPA.rae->moveToAddress(theta + offset);
  offset += modelPPA.rae->getThetaSize();
  modelPPB.rae->moveToAddress(theta + offset);
  offset += modelPPB.rae->getThetaSize();
  modelA.rae->de_->moveToAddress(theta + offset);
  offset += modelA.rae->de_->getThetaSize();
  assert ( offset == ab_size );

  // Sets some model parameters such as maximum sentence and node length to
  // speed up memory access and parallelisation later on.
  modelA.finalize();
  modelB.finalize();
  modelPPA.finalize();
  modelPPB.finalize();

  modelA.bools = bools1;
  modelA.alpha = alpha;
  modelA.gamma = gamma;

  modelA.from = 0;
  modelA.to = modelA.corpus.size();
  // if (config.num_sentences != 0)
  // {
    // cout << "Reset to small" << endl;
    // modelA.to = min(int(modelA.corpus.size()),config.num_sentences);
  // }
  modelA.num_noise_samples = vm["noise"].as<int>();

  modelB.bools = bools2;
  modelB.alpha = alpha;
  modelB.gamma = gamma;
  modelB.from = 0;
  modelB.to = modelB.corpus.size();
  modelB.num_noise_samples = vm["noise"].as<int>();

  modelPPA.bools = bools1;
  modelPPA.alpha = alpha;
  modelPPA.gamma = gamma;
  modelPPA.from = 0;
  modelPPA.to = modelPPA.corpus.size();
  modelPPA.num_noise_samples = vm["noise"].as<int>();

  modelPPB.bools = bools2;
  modelPPB.alpha = alpha;
  modelPPB.gamma = gamma;
  modelPPB.from = 0;
  modelPPB.to = modelPPB.corpus.size();
  modelPPB.num_noise_samples = vm["noise"].as<int>();

  cout << "Dict size: " << modelA.rae->getDictSize() << " and " << modelB.rae->getDictSize() << endl;

  modelA.trainer = new OpenQABordesTrainer();

  /***************************************************************************
   *                              BFGS training                              *
   ***************************************************************************/

  modelA.b = &modelB; // magic setting: merge the models for propagation
  modelA.docmod = &modelPPA;
  modelB.docmod = &modelPPB;

  if (config.training_method == 0)
  {
    cout << "Training with LBFGS" << endl;
    // train_lbfgs(modelA,linesearch,iterations,epsilon,lambdas);
  }
  else if (config.training_method == 1)
  {
    cout << "Training with SGD" << endl;
    train_sgd(modelA,iterations,eta,lambdas);
  }
  else if (config.training_method == 2)
  {
    cout << "Finite Gradient Check" << endl;
    // train_adagrad(modelA,3,eta,nullptr,batches,lambdas,l1);
    finite_quad_check(modelA,lambdas);
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
    // train_lbfgs_minibatch(modelA,linesearch,iterations,epsilon,batches,lambdas);
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
