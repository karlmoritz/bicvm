// File: finite_grad_check.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 09-01-2013
// Last Update: Thu 29 May 2014 11:30:31 BST

#ifndef COMMON_FINITE_GRAD_CHECK_H
#define COMMON_FINITE_GRAD_CHECK_H

// Local
#include "shared_defs.h"

void finite_grad_check(Model &model, Lambdas lambdas);
void finite_bigrad_check(Model &model, Lambdas lambdas);
void finite_quad_check(Model &model, Lambdas lambdas);

#endif  // COMMON_FINITE_GRAD_CHECK_H

