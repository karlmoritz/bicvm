// File: finite_grad_check.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 09-01-2013
// Last Update: Thu 17 Oct 2013 03:39:20 PM BST
/*------------------------------------------------------------------------
 * Description: <DESC>
 *
 *------------------------------------------------------------------------
 * History:
 * TODO:
 *========================================================================
 */

#ifndef FINITE_GRAD_CHECK_H_EPMOQDQN
#define FINITE_GRAD_CHECK_H_EPMOQDQN

// Local
#include "shared_defs.h"

void finite_grad_check(Model &model, Lambdas lambdas);
void finite_bigrad_check(Model &model, Lambdas lambdas);

#endif /* end of include guard: FINITE_GRAD_CHECK_H_EPMOQDQN */

