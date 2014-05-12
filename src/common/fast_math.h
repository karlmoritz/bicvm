// File: fast_math.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 23-01-2013
// Last Update: Fri 11 Oct 2013 03:53:12 PM BST
/*------------------------------------------------------------------------
 * Description: <DESC>
 *
 *------------------------------------------------------------------------
 * History:
 * TODO:
 *========================================================================
 */

#include <vector>

#include "shared_defs.h"
#include "../utils/fastonebigheader.h"

#if   LBFGS_FLOAT == 32

static inline float getSigmoid(float x) {
  return 1.0f / (1.0f + fasterexp (-x));
}
static inline float getExp(Real p) {
  return fasterpow2 (1.442695040f * p);
}
static inline float getTanh(Real p) {
  return -1.0f + 2.0f / (1.0f + fasterexp (-2.0f * p));
}

#elif LBFGS_FLOAT == 64

static inline float getSigmoid(float x) {
  return 1.0f / (1.0f + exp (-x));
}
static inline float getExp(Real p) {
  return exp(p);
}
static inline float getTanh(Real p) {
  return tanh(p);
}

#endif

Real getArgtanh(Real val);

Real getSpearmansRho(std::vector<Real> s1, std::vector<Real> s2);
std::vector<Real> getSplitRank(std::vector<Real>& v);

MatrixReal tanh_p(VectorReal& x); // takes unnormalized tan vector

Real rectifiedLinear(Real val);
Real rectifiedLinearGrad(Real val);

template <typename T> inline constexpr
int signum(T x, std::false_type is_signed) {
      return T(0) < x;
}

template <typename T> inline constexpr
int signum(T x, std::true_type is_signed) {
      return (T(0) < x) - (x < T(0));
}

template <typename T> inline constexpr
int signum(T x) {
      return signum(x, std::is_signed<T>());
}
