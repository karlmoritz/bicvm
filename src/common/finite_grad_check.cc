// File: finite_grad_check.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 01-01-2013
// Last Update: Tue 07 Jan 2014 10:58:55 PM GMT
/*------------------------------------------------------------------------
 * Description: <DESC>
 *
 *------------------------------------------------------------------------
 * History:
 * TODO:
 *========================================================================
 */

// STL
#include <iostream>
#include <cmath>

// Boost
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>

#include "shared_defs.h"

// L-BFGS
#include <lbfgs.h>

// Local
#include "finite_grad_check.h"
#include "train_update.h"
#include "recursive_autoencoder.h"

using namespace std;
namespace bpo = boost::program_options;


void finite_grad_check(Model &model, Lambdas lambdas)
{
  Real* vars = nullptr;
  int number_vars = 0;

  modvars<int> counts;
  model.rae->setIncrementalCounts(&counts, vars, number_vars);
  model.lambdas = lambdas;

  WeightVectorType theta(vars,number_vars);

  int extended_vars = number_vars;
  if (model.docmod != nullptr) {
    extended_vars += model.docmod->rae->getThetaSize();
  }
  Real* data1 = new Real[extended_vars]();
  Real* data2 = new Real[extended_vars]();
  WeightVectorType grad(data1,number_vars);

  BProps props(model);
  Real error1 = 0.0, error2;
  computeCostAndGrad(model,nullptr,data1,number_vars,0,props,&error1);

  modvars<Real> dists;
  dists.init();
  Real dist = 0.0;
  Real delta = 1.0e-7;

  for (int i=0;i<number_vars;++i)
  {
    theta[i] += delta;
    error2 = 0.0;
    computeCostAndGrad(model,nullptr,data2,number_vars,0,props,&error2);
    Real xdev = (error2 - error1) / delta;
    theta[i] -= delta;
    if (i < counts.D)   { dists.D += abs(grad[i] - xdev);   cout << "D   "; }
    else if (i < counts.U)  { dists.U += abs(grad[i] - xdev);  cout << "U  "; }
    else if (i < counts.V)  { dists.V += abs(grad[i] - xdev);  cout << "V  "; }
    else if (i < counts.W)  { dists.W += abs(grad[i] - xdev);  cout << "W  "; }
    else if (i < counts.A)  { dists.A += abs(grad[i] - xdev);  cout << "A  "; }
    else if (i < counts.Wd)  { dists.Wd += abs(grad[i] - xdev);  cout << "Wd  "; }
    else if (i < counts.Wdr) { dists.Wdr += abs(grad[i] - xdev); cout << "Wdr "; }
    else if (i < counts.Bd)  { dists.Bd += abs(grad[i] - xdev);  cout << "Bd  "; }
    else if (i < counts.Bdr) { dists.Bdr += abs(grad[i] - xdev); cout << "Bdr "; }
    else if (i < counts.Wf)  { dists.Wf += abs(grad[i] - xdev);  cout << "Wf  "; }
    else if (i < counts.Wl)  { dists.Wl += abs(grad[i] - xdev);  cout << "Wl  "; }
    else if (i < counts.Bl)  { dists.Bl += abs(grad[i] - xdev);  cout << "Bl  "; }

    //template<> void modvars<int>::init() { D = 0; U = 0; V = 0; W = 0; A = 0; Wd = 0; Wdr = 0; Bd = 0; Bdr = 0; Wf = 0; Wl = 0; Bl = 0; alpha_rae = 0; alpha_lbl = 0; }
    cout << i << ": " << grad[i] << " .. " << " vs " << (xdev) << "   " << error2 << " - " << error1 << "[" << theta[i] << "]" << endl;
    cout << "Xdev: " << error1 / delta << " vs " << error2 / delta << endl;

    dist += abs(grad[i] - xdev);
  }

  cout << "total: " << dist << " D/U/V/W/A/Wd/Wdr/Bd/Bdr/Wf/Wl/Bl " << endl;
  cout << dists.D << " ";
  cout << dists.U << " ";
  cout << dists.V << " ";
  cout << dists.W << " ";
  cout << dists.A << " ";
  cout << dists.Wd << " " << dists.Wdr << " " << dists.Bd << " " << dists.Bdr << " ";
  cout << dists.Wf << " ";
  cout << dists.Wl << " " << dists.Bl << endl;
  // Don't care about what's next
  assert(false);
}

void finite_bigrad_check(Model &model, Lambdas lambdas)
{
  Real* vars = nullptr;
  Real* varsX = nullptr;

  int number_vars = 0;
  modvars<int> counts;
  model.rae->setIncrementalCounts(&counts, vars, number_vars);
  model.lambdas = lambdas;

  int number_varsB = 0;
  modvars<int> countsB;
  model.b->rae->setIncrementalCounts(&countsB, varsX, number_varsB);

  int double_vars = number_vars + number_varsB;

  // cout.precision(15);

  WeightVectorType theta(vars,double_vars);
  int extended_vars = double_vars;
  if (model.docmod != nullptr) {
    extended_vars += model.docmod->rae->getThetaSize() +
      model.b->docmod->rae->getThetaSize();
  }
  Real* data1 = new Real[extended_vars]();
  Real* data2 = new Real[extended_vars]();
  Real* data3 = new Real[extended_vars]();
  WeightVectorType grad(data1,double_vars);

  Real* theta2 = new Real[double_vars]();
  WeightVectorType t2(theta2,double_vars);
  WeightVectorType t1(vars,double_vars);
  t2 = t1;

  cout << "Theta size: " << model.rae->getThetaSize() << endl;
  Real error1 = 0.0, error2;
#pragma omp parallel
  {
  BProps props(model);
      Real errorX = 0;
  computeCostAndGrad(model,nullptr,data1,double_vars,0,props,&errorX);
#pragma omp critical
      {
        error1 += errorX;
      }
  }
  for (auto i=0; i < extended_vars; ++i) data3[i] = data1[i];
  // Run twice because of docgrad thingy.
  error1 = 0.0;
#pragma omp parallel
  {
  BProps props(model);
      Real errorX = 0;
  computeCostAndGrad(model,nullptr,data1,double_vars,1,props,&errorX);
#pragma omp critical
      {
        error1 += errorX;
      }
  }

  modvars<Real> dists;
  modvars<Real> distsB;
  dists.init();
  distsB.init();

  Real dist = 0.0;

  Real delta = 0.001;
  int j = 0;

  for (int i=0;i<double_vars;++i) {
    t1 = t2;
    theta[i] += delta;
    for (auto i=0; i < extended_vars; ++i) data2[i] = data3[i];
#pragma omp parallel
    {
      BProps props(model);
      Real errorX = 0;
      computeCostAndGrad(model,nullptr,data2,double_vars,0,props,&errorX);
#pragma omp critical
      {
        error2 += errorX;
      }

    }
    error2 = 0.0;
#pragma omp parallel
    {
      BProps props(model);
      Real errorX = 0;
      computeCostAndGrad(model,nullptr,data2,double_vars,1,props,&errorX);
#pragma omp critical
      {
        error2 += errorX;
      }
    }
    // Real xdev = (error2 - error1) / delta;
    Real xdev = (Real)((error2 - error1) / delta);
    theta[i] -= delta;
    j = i - number_vars;
    if (i < counts.D)   { dists.D += abs(grad[i] - xdev);   cout << "D   "; }
    else if (i < counts.U)  { dists.U += abs(grad[i] - xdev);  cout << "U  "; }
    else if (i < counts.V)  { dists.V += abs(grad[i] - xdev);  cout << "V  "; }
    else if (i < counts.W)  { dists.W += abs(grad[i] - xdev);  cout << "W  "; }
    else if (i < counts.A)  { dists.A += abs(grad[i] - xdev);  cout << "A  "; }
    else if (i < counts.Wd)  { dists.Wd += abs(grad[i] - xdev);  cout << "Wd  "; }
    else if (i < counts.Wdr) { dists.Wdr += abs(grad[i] - xdev); cout << "Wdr "; }
    else if (i < counts.Bd)  { dists.Bd += abs(grad[i] - xdev);  cout << "Bd  "; }
    else if (i < counts.Bdr) { dists.Bdr += abs(grad[i] - xdev); cout << "Bdr "; }
    else if (i < counts.Wf)  { dists.Wf += abs(grad[i] - xdev);  cout << "Wf  "; }
    else if (i < counts.Wl)  { dists.Wl += abs(grad[i] - xdev);  cout << "Wl  "; }
    else if (i < counts.Bl)  { dists.Bl += abs(grad[i] - xdev);  cout << "Bl  "; }

    else if (j < countsB.D)   { distsB.D += abs(grad[i] - xdev);   cout << "2D   "; }
    else if (j < countsB.U)  { distsB.U += abs(grad[i] - xdev);  cout << "2U  "; }
    else if (j < countsB.V)  { distsB.V += abs(grad[i] - xdev);  cout << "2V  "; }
    else if (j < countsB.W)  { distsB.W += abs(grad[i] - xdev);  cout << "2W  "; }
    else if (j < countsB.A)  { distsB.A += abs(grad[i] - xdev);  cout << "2A  "; }
    else if (j < countsB.Wd)  { distsB.Wd += abs(grad[i] - xdev);  cout << "2Wd  "; }
    else if (j < countsB.Wdr) { distsB.Wdr += abs(grad[i] - xdev); cout << "2Wdr "; }
    else if (j < countsB.Bd)  { distsB.Bd += abs(grad[i] - xdev);  cout << "2Bd  "; }
    else if (j < countsB.Bdr) { distsB.Bdr += abs(grad[i] - xdev); cout << "2Bdr "; }
    else if (j < countsB.Wf)  { distsB.Wf += abs(grad[i] - xdev);  cout << "2Wf  "; }
    else if (j < countsB.Wl)  { distsB.Wl += abs(grad[i] - xdev);  cout << "2Wl  "; }
    else if (j < countsB.Bl)  { distsB.Bl += abs(grad[i] - xdev);  cout << "2Bl  "; }
    //template<> void modvars<int>::init() { D = 0; U = 0; V = 0; W = 0; A = 0; Wd = 0; Wdr = 0; Bd = 0; Bdr = 0; Wf = 0; Wl = 0; Bl = 0; alpha_rae = 0; alpha_lbl = 0; }
    cout << i << ": " << grad[i] << " vs " << (xdev) << "   " << error2 << " - " << error1 << endl;

    dist += abs(grad[i] - xdev);
  }

  cout << "total: " << dist << " D/U/V/W/A/Wd/Wdr/Bd/Bdr/Wf/Wl/Bl " << endl;
  cout << dists.D << " ";
  cout << dists.U << " ";
  cout << dists.V << " ";
  cout << dists.W << " ";
  cout << dists.A << " ";
  cout << dists.Wd << " ";
  cout << dists.Wdr << " ";
  cout << dists.Bd << " ";
  cout << dists.Bdr << " ";
  cout << dists.Wf << " ";
  cout << dists.Wl << " ";
  cout << dists.Bl << endl;

  cout << distsB.D << " ";
  cout << distsB.U << " ";
  cout << distsB.V << " ";
  cout << distsB.W << " ";
  cout << distsB.A << " ";
  cout << distsB.Wd << " ";
  cout << distsB.Wdr << " ";
  cout << distsB.Bd << " ";
  cout << distsB.Bdr << " ";
  cout << distsB.Wf << " ";
  cout << distsB.Wl << " ";
  cout << distsB.Bl << endl;
  // Don't care about what's next
  assert(false);
}

