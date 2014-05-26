// File: load_qqpair.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 22-01-2013
// Last Update: Mon 26 May 2014 14:40:33 BST

#ifndef COMMON_LOAD_QQPAIR_H
#define COMMON_LOAD_QQPAIR_H

#include "shared_defs.h"
#include "recursive_autoencoder.h"
#include "senna.h"

namespace load_qqpair {

void load_file(TrainingCorpus& corpusA,
               TrainingCorpus& corpusB,
               TrainingCorpus& corpusC,
               string file_name,
               bool create_dict,
               bool remove_type_ending,
               Senna& senna);

Sentence createSentence(Senna& senna, vector<string>& words, int truth_label=0);

}
#endif  // COMMON_LOAD_QQPAIR_H
