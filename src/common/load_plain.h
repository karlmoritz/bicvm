// File: load_plain.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 22-01-2013
// Last Update: Thu 17 Oct 2013 04:12:31 PM BST

#ifndef COMMON_LOAD_PLAIN_H
#define COMMON_LOAD_PLAIN_H

#include "../pugi/pugixml.hpp"

#include "shared_defs.h"
#include "recursive_autoencoder.h"
#include "senna.h"

using namespace pugi;

namespace load_plain {

void load(TrainingCorpus& trainCorpus, TrainingCorpus& testCorpus, string
    file_positive, string file_negative, bool add_to_dict, Senna& senna);

void load_file(TrainingCorpus& corpus, string file_name,
     int label, bool add_to_dict,
    Senna& senna);

Sentence createSentence(Senna& senna, vector<string>& words, int label, bool add_to_dict=false);

}
#endif /* COMMON_LOAD_PLAIN_H */
