// File: shared_defs.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 07-01-2013
// Last Update: Fri 30 May 2014 15:21:59 BST

#ifndef COMMON_SHARED_DEFS_H
#define COMMON_SHARED_DEFS_H

// STL
#include <exception>
#include <iostream>
#include <boost/assign.hpp>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Core>

#include "grammarrules.h"

using namespace std;

/***************************************************************************
 *                                Typedefs                                 *
 ***************************************************************************/
// #ifndef LBFGS_FLOAT
// #define LBFGS_FLOAT 32
// #endif

/* #if   LBFGS_FLOAT == 32 */
/* typedef float Real; // necessary for fast math library */
/* #elif LBFGS_FLOAT == 64 */
/* typedef double Real; */
/* #else */
typedef float Real;
/* #endif */

typedef Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixReal;
typedef Eigen::Matrix<Real, Eigen::Dynamic, 1>              VectorReal;
typedef Eigen::Array<Real, Eigen::Dynamic, 1>               ArrayReal;

typedef Eigen::Map<MatrixReal>            WeightMatrixType;
typedef std::vector<WeightMatrixType>     WeightMatricesType;
typedef Eigen::Map<VectorReal>            WeightVectorType;
typedef Eigen::Map<ArrayReal>             WeightArrayType;
typedef std::vector<WeightVectorType>     WeightVectorsType;

typedef std::map<std::pair<int,int>, int> I2Map;
typedef std::map<std::pair<int,std::pair<int,int>>, int> I3Map;

enum TreeType {
  TREE_FOREST,
  TREE_CCG,
  TREE_STANFORD,
  TREE_ALPINO, // ~= Stanford but different format
  TREE_PLAIN
};

enum LineSearchType { // Matching numbering from lbfgs
  MORETHUENTE = 0,
  WOLFE = 2,
  ARMIJO = 1,
  STRONGWOLFE = 3
};

//("backtracking",WOLFE)
static std::map<string,LineSearchType> s2line_map =
boost::assign::map_list_of
("morethuente",MORETHUENTE)
("wolfe",WOLFE)
("armijo",ARMIJO)
("strongwolfe",STRONGWOLFE);

#include "dictionary.h"

struct Sentence
{
  int             value;
  Sentence(int val) : value(val) {}
  vector<LabelID> words;
  vector<int>     child0;
  vector<int>     child1;
  vector<int>     nodes; // -1 if treenode, >=0: pointer to word in words
  vector<CCGRuleType>     rule;
  vector<int>             cat; // = POS / phrase level tag
  vector<int>             tree_size;
  int id;     // unique ID of this sentence.
  int doc_id; // unique ID of the document this sentence belongs to (not strictly necessary?)
};

// Forward definition
class RecursiveAutoencoderBase;
class BackpropagatorBase;
class Trainer;

typedef vector<Sentence> TrainingCorpus;

template <typename T>
struct modvars
{
  T D;
  T U;
  T V;
  T W;
  T A;
  T Wd;
  T Wdr;
  T Bd;
  T Bdr;
  T Wf;
  T Wl;
  T Bl;

  T alpha_rae;
  T alpha_lbl;

  modvars() { init(); }
  void init() {};
  void multiply(Real multiplier) {};
};

template<> inline void modvars<float>::init() { D = 0.0; U = 0.0; V = 0.0; W = 0.0; A = 0.0; Wd = 0.0; Wdr = 0.0; Bd = 0.0; Bdr = 0.0; Wf = 0.0; Wl = 0.0; Bl = 0.0; alpha_rae = 0.0; alpha_lbl = 0.0; }
template<> inline void modvars<double>::init() { D = 0.0; U = 0.0; V = 0.0; W = 0.0; A = 0.0; Wd = 0.0; Wdr = 0.0; Bd = 0.0; Bdr = 0.0; Wf = 0.0; Wl = 0.0; Bl = 0.0; alpha_rae = 0.0; alpha_lbl = 0.0; }
template<> inline void modvars<int>::init() { D = 0; U = 0; V = 0; W = 0; A = 0; Wd = 0; Wdr = 0; Bd = 0; Bdr = 0; Wf = 0; Wl = 0; Bl = 0; alpha_rae = 0; alpha_lbl = 0; }
template<> inline void modvars<bool>::init() { D = true; U = true; V = true; W = true; A = true; Wd = true; Wdr = true; Bd = true; Bdr = true; Wf = true; Wl = true; Bl = true; alpha_rae = true; alpha_lbl = true; }

template<> inline void modvars<Real>::multiply(Real multiplier) {
  D *= multiplier; U *= multiplier; V *= multiplier; W *= multiplier; A *= multiplier; Wd *= multiplier; Wdr *= multiplier; Bd *= multiplier; Bdr *= multiplier; Wf *= multiplier; Wl *= multiplier; Bl *= multiplier; alpha_rae *= multiplier; alpha_lbl *= multiplier;
}

typedef modvars<Real> Lambdas;
typedef modvars<bool>  Bools;
typedef modvars<int>   Counts;

struct Model
{
  TrainingCorpus corpus;
  std::vector<int> indexes;
  RecursiveAutoencoderBase* rae;
  Lambdas lambdas;
  Real alpha;
  Real beta;
  Real gamma; // used for down-weighting nc error
  Bools bools;

  int normalization_type;

  // Minibatch variables
  int from;
  int to;

  Model* b; // Use this if models should be trained jointly
  Model* a; // Use this if model should only use a.fProp as measure

  Trainer* trainer; // the trainer module (openqa or general). Used to call
  // computecostandgrad function.

  // docmod is a pointer to a higher model that treats the sentences from this
  // model as words and the documents from this model as sentences.
  Model* docmod;
  int it_count;

  // Really dirty data structure abusre
  int num_noise_samples;
  int noise_sample_offset;

  // L2?
  bool calc_L2;
  // bool calc_L1;

  int max_sent_length;
  int max_node_length;

  WeightVectorsType vectorsA; // Use this if biprop should use vectorsA as target vectors
  Model(RecursiveAutoencoderBase* rae_, TrainingCorpus corp);
  // Model(RecursiveAutoencoderBase& rae_);
  Model();
  void finalize();
};

struct BProps
{
  BackpropagatorBase* propA;
  BackpropagatorBase* propB;
  BProps* docprop;
  BProps(const Model& a);
  BProps(const Model& a, const Model& b);
  BProps(const Model& a, const Model& b, const Model& c, const Model& d);

  BProps(const Model& a, bool share_dict);
  BProps(const Model& a, const Model& b, Real* dictptr);
};

struct ENotImplemented : public exception
{
  const char * what () const throw ()
  {
    return "Functionality not implemented/available here.";
  }
};

#endif  // COMMON_SHARED_DEFS_H
