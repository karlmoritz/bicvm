// File: load_relent.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 22-01-2013
// Last Update: Tue 20 May 2014 19:09:58 BST

#ifndef COMMON_LOAD_RELENT_H
#define COMMON_LOAD_RELENT_H

#include "shared_defs.h"
#include "recursive_autoencoder.h"
#include "senna.h"

namespace load_relent {

void load_file(TrainingCorpus& corpus,
               string file_name,
               bool create_dict,
               Senna& senna);

Sentence createSentence(Senna& senna, vector<string>& words, bool add_to_dict);

}
#endif  // COMMON_LOAD_RELENT_H
