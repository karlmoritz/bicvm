// File: finite_grad_check.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 01-01-2013
// Last Update: Thu 29 May 2014 13:41:02 BST

// STL
#include <iostream>
#include <cmath>

// Boost
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>

#include "shared_defs.h"

// Local
#include "finite_grad_check.h"
#include "trainer.h"
#include "general_trainer.h"
#include "openqa_trainer.h"
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

  int number_dA = model.rae->getThetaDSize();
  int all_vars = number_vars + number_dA;

  WeightVectorType theta(vars,all_vars);

  int extended_vars = number_vars;
  if (model.docmod != nullptr) {
    extended_vars += model.docmod->rae->getThetaSize();
    extended_vars += model.docmod->rae->getThetaDSize();
  }
  Real* data1 = new Real[extended_vars]();
  Real* data2 = new Real[extended_vars]();
  WeightVectorType grad(data1,number_vars);

  BProps props(model);
  Real error1 = 0.0, error2;
  model.trainer->computeCostAndGrad(model,nullptr,data1,all_vars,0,props,&error1);

  modvars<Real> dists;
  dists.init();
  Real dist = 0.0;
  Real delta = 1.0e-7;

  for (int i=0;i<all_vars;++i)
  {
    theta[i] += delta;
    error2 = 0.0;
    model.trainer->computeCostAndGrad(model,nullptr,data2,all_vars,0,props,&error2);
    Real xdev = (error2 - error1) / delta;
    theta[i] -= delta;
    if (i < counts.U)  { dists.U += abs(grad[i] - xdev);  cout << "U  "; }
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
    else if (i < (number_vars + number_dA))   { dists.D += abs(grad[i] - xdev);   cout << "D   "; }
    else  { cout << "ERROR  "; assert(false); }

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
  // Find space for all variables.
  Real* vars = nullptr;
  Real* varsX = nullptr;

  int number_vars = 0;
  modvars<int> counts;
  model.rae->setIncrementalCounts(&counts, vars, number_vars);
  model.lambdas = lambdas;

  int number_varsB = 0;
  modvars<int> countsB;
  model.b->rae->setIncrementalCounts(&countsB, varsX, number_varsB);

  int number_dA = model.rae->getThetaDSize();
  int number_dB = model.b->rae->getThetaDSize();
  int double_vars = number_vars + number_varsB + number_dA + number_dB;

  WeightVectorType theta(vars,double_vars);
  int extended_vars = double_vars;
  if (model.docmod != nullptr) {
    extended_vars += model.docmod->rae->getThetaSize() +
      model.docmod->rae->getThetaDSize() +
      model.b->docmod->rae->getThetaSize() +
      model.b->docmod->rae->getThetaDSize();
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
  model.trainer->computeCostAndGrad(model,nullptr,data1,double_vars,0,props,&errorX);
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
  model.trainer->computeCostAndGrad(model,nullptr,data1,double_vars,1,props,&errorX);
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
    for (auto r=0; r < extended_vars; ++r) data2[r] = data3[r];
#pragma omp parallel
    {
      BProps props(model);
      Real errorX = 0;
      model.trainer->computeCostAndGrad(model,nullptr,data2,double_vars,0,props,&errorX);
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
      model.trainer->computeCostAndGrad(model,nullptr,data2,double_vars,1,props,&errorX);
#pragma omp critical
      {
        error2 += errorX;
      }
    }
    // Real xdev = (error2 - error1) / delta;
    Real xdev = (Real)((error2 - error1) / delta);
    theta[i] -= delta;
    j = i - number_vars;
    if (i < counts.U)  { dists.U += abs(grad[i] - xdev);  cout << "U  "; }
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

    else if (j < (number_varsB + number_dA))   { dists.D += abs(grad[i] - xdev);   cout << "D   "; }
    else if (j < (number_varsB + number_dA + number_dB))   { distsB.D += abs(grad[i] - xdev);   cout << "2D  "; }

    else  { cout << "ERROR  " << endl; assert(false); }
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

void finite_quad_check(Model &model, Lambdas lambdas)
{
  // Find space for all variables.
  Real* theta_ptr = nullptr;
  Real* tmp = nullptr;

  model.lambdas = lambdas;

  int number_vars_A = 0;
  modvars<int> counts_A;
  model.rae->setIncrementalCounts(&counts_A, theta_ptr, number_vars_A);

  int number_vars_B = 0;
  modvars<int> counts_B;
  model.b->rae->setIncrementalCounts(&counts_B, tmp, number_vars_B);

  int number_vars_C = 0;
  modvars<int> counts_C;
  model.b->docmod->rae->setIncrementalCounts(&counts_C, tmp, number_vars_C);

  int number_vars_D = 0;
  modvars<int> counts_D;
  model.docmod->rae->setIncrementalCounts(&counts_D, tmp, number_vars_D);

  int number_dict = model.rae->getThetaDSize();

  int model_vars = number_vars_A + number_vars_B + number_vars_C + number_vars_D
    + number_dict;

  WeightVectorType theta(theta_ptr, model_vars);  // model variables.

  Real* data1 = new Real[model_vars]();  // storage for gradients from vanilla run
  Real* data2 = new Real[model_vars]();  // spaceholder for subsequent gradients
  WeightVectorType grad(data1,model_vars);  // gradient location

  Real error1 = 0.0, error2 = 0.0;
#pragma omp parallel
  {
    BProps props(model);
    Real errorX = 0;
    model.trainer->computeCostAndGrad(model,nullptr,data1,model_vars,0,props,&errorX);
#pragma omp critical
    {
      error1 += errorX;
    }
  }

  // Divergences between model and true gradient.
  modvars<Real> dists_A, dists_B, dists_C, dists_D;
  dists_A.init();
  dists_B.init();
  dists_C.init();
  dists_D.init();

  Real dist = 0.0;
  Real delta = 0.001;
  int j = 0;

  for (int i=0; i<model_vars; ++i) {
    theta[i] += delta;  // Increase by delta.
    error2 = 0.0;
#pragma omp parallel
    {
      BProps props(model);
      Real errorX = 0;
      model.trainer->computeCostAndGrad(model,nullptr,data2,model_vars,0,props,&errorX);
#pragma omp critical
      {
        error2 += errorX;
      }
    }
    Real xdev = (Real)((error2 - error1) / delta);
    theta[i] -= delta;  // Reset model variables to original state.
    j = i - number_vars_A;
    if (i < counts_A.U)  { dists_A.U += abs(grad[i] - xdev);  cout << "U  "; }
    else if (i < counts_A.V)  { dists_A.V += abs(grad[i] - xdev);  cout << "V  "; }
    else if (i < counts_A.W)  { dists_A.W += abs(grad[i] - xdev);  cout << "W  "; }
    else if (i < counts_A.A)  { dists_A.A += abs(grad[i] - xdev);  cout << "A  "; }
    else if (i < counts_A.Wd)  { dists_A.Wd += abs(grad[i] - xdev);  cout << "Wd  "; }
    else if (i < counts_A.Wdr) { dists_A.Wdr += abs(grad[i] - xdev); cout << "Wdr "; }
    else if (i < counts_A.Bd)  { dists_A.Bd += abs(grad[i] - xdev);  cout << "Bd  "; }
    else if (i < counts_A.Bdr) { dists_A.Bdr += abs(grad[i] - xdev); cout << "Bdr "; }
    else if (i < counts_A.Wf)  { dists_A.Wf += abs(grad[i] - xdev);  cout << "Wf  "; }
    else if (i < counts_A.Wl)  { dists_A.Wl += abs(grad[i] - xdev);  cout << "Wl  "; }
    else if (i < counts_A.Bl)  { dists_A.Bl += abs(grad[i] - xdev);  cout << "Bl  "; }

    else if (j < counts_B.U)  { dists_B.U += abs(grad[i] - xdev);  cout << "2U  "; }
    else if (j < counts_B.V)  { dists_B.V += abs(grad[i] - xdev);  cout << "2V  "; }
    else if (j < counts_B.W)  { dists_B.W += abs(grad[i] - xdev);  cout << "2W  "; }
    else if (j < counts_B.A)  { dists_B.A += abs(grad[i] - xdev);  cout << "2A  "; }
    else if (j < counts_B.Wd)  { dists_B.Wd += abs(grad[i] - xdev);  cout << "2Wd  "; }
    else if (j < counts_B.Wdr) { dists_B.Wdr += abs(grad[i] - xdev); cout << "2Wdr "; }
    else if (j < counts_B.Bd)  { dists_B.Bd += abs(grad[i] - xdev);  cout << "2Bd  "; }
    else if (j < counts_B.Bdr) { dists_B.Bdr += abs(grad[i] - xdev); cout << "2Bdr "; }
    else if (j < counts_B.Wf)  { dists_B.Wf += abs(grad[i] - xdev);  cout << "2Wf  "; }
    else if (j < counts_B.Wl)  { dists_B.Wl += abs(grad[i] - xdev);  cout << "2Wl  "; }
    else if (j < counts_B.Bl)  { dists_B.Bl += abs(grad[i] - xdev);  cout << "2Bl  "; }

    else if (j < (number_vars_B + number_dict))   { dists_A.D += abs(grad[i] - xdev);   cout << "D   "; }
    // CLEAN THIS UP WITH NUMBERS FOR THE OTHER TWO MODELS etc.
    else  { dists_A.D += abs(grad[i] - xdev);   cout << "D   "; }
    // else if (j < (number_vars_B + number_dA + number_dB))   { dists_B.D += abs(grad[i] - xdev);   cout << "2D  "; }

    // else  { cout << "ERROR  " << endl; assert(false); }
    cout << i << ": (observed)\t" << grad[i] << " vs " << (xdev) << "\t(Error-diff)\t" << error1 << "\t" << error2 << endl;

    dist += abs(grad[i] - xdev);
  }

  cout << "total: " << dist << " D/U/V/W/A/Wd/Wdr/Bd/Bdr/Wf/Wl/Bl " << endl;
  cout << dists_A.D << " ";
  cout << dists_A.U << " ";
  cout << dists_A.V << " ";
  cout << dists_A.W << " ";
  cout << dists_A.A << " ";
  cout << dists_A.Wd << " ";
  cout << dists_A.Wdr << " ";
  cout << dists_A.Bd << " ";
  cout << dists_A.Bdr << " ";
  cout << dists_A.Wf << " ";
  cout << dists_A.Wl << " ";
  cout << dists_A.Bl << endl;

  cout << dists_B.D << " ";
  cout << dists_B.U << " ";
  cout << dists_B.V << " ";
  cout << dists_B.W << " ";
  cout << dists_B.A << " ";
  cout << dists_B.Wd << " ";
  cout << dists_B.Wdr << " ";
  cout << dists_B.Bd << " ";
  cout << dists_B.Bdr << " ";
  cout << dists_B.Wf << " ";
  cout << dists_B.Wl << " ";
  cout << dists_B.Bl << endl;
  // Don't care about what's next
  assert(false);
}

