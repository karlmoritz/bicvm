// File: train_lbfgs.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 09-01-2013
// Last Update: Fri 11 Oct 2013 03:53:14 PM BST

#ifndef COMMON_TRAIN_LBFGS_H
#define COMMON_TRAIN_LBFGS_H

// L-BFGS
#include <lbfgs.h>

// Local
#include "shared_defs.h"

static int progress(
    void *instance,
    const Real *x,
    const Real *g,
    const Real fx,
    const Real xnorm,
    const Real gnorm,
    const Real step,
    int n,
    int k,
    int ls
    );

static int progress_minibatch(
    void *instance,
    const Real *x,
    const Real *g,
    const Real fx,
    const Real xnorm,
    const Real gnorm,
    const Real step,
    int n,
    int k,
    int ls
    );


Real evaluate_(
    void *instance,
    const Real *x, // Variables (theta)
    Real *g,       // Put gradient here
    const int n,              // Number of variables
    const Real step);  // line-search step used in this iteration

int train_lbfgs(Model& model, LineSearchType linesearch, int max_iterations, Real epsilon, Lambdas lambdas);

int train_lbfgs_minibatch(Model& model, LineSearchType linesearch, int
    max_iterations, Real epsilon, int batches, Lambdas lambdas);

#endif  // COMMON_TRAIN_LBFGS_H
