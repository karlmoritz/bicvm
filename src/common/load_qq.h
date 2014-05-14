// File: load_qq.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 22-01-2013
// Last Update: Wed 14 May 2014 16:02:06 BST

#ifndef COMMON_LOAD_QQ_H
#define COMMON_LOAD_QQ_H

#include "../pugi/pugixml.hpp"

#include "shared_defs.h"
#include "recursive_autoencoder.h"
#include "senna.h"

using namespace pugi;

namespace load_qq {

void load_file(TrainingCorpus& corpusA,
               TrainingCorpus& corpusB,
               string file_name,
               bool create_dict_A,
               bool create_dict_B,
               Senna& sennaA,
               Senna& sennaB);

Sentence createSentence(Senna& senna, vector<string>& words, bool add_to_dict);

}
#endif  // COMMON_LOAD_QQ_H
