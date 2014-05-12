// File: senna.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 08-02-2013
// Last Update: Fri 11 Oct 2013 03:53:14 PM BST
/*------------------------------------------------------------------------
 * Description: <DESC>
 *
 *------------------------------------------------------------------------
 * History:
 * TODO:
 *========================================================================
 */

#ifndef SENNA_H_RI0HO1A6
#define SENNA_H_RI0HO1A6

#include "dictionary.h"

class RecursiveAutoencoderBase;

class Senna
{

public:
  Senna(RecursiveAutoencoderBase& rae, int embeddings_type);
  void applyEmbeddings();
  LabelID id(const Label& label);
  LabelID id(const Label& l, bool add_new=false);

  int good_counter;
  int bad_counter;

private:
  map<string,int> words;
  map<int, vector<Real> > embeddings;
  RecursiveAutoencoderBase& rae;
  bool use_embeddings;
};

#endif /* end of include guard:  */
