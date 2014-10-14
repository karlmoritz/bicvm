// File: load_doc.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 22-01-2013
// Last Update: Thu 02 Jan 2014 12:08:08 PM GMT

#ifndef COMMON_LOAD_DOC_H
#define COMMON_LOAD_DOC_H

#include "../pugi/pugixml.hpp"

#include "shared_defs.h"
#include "recursive_autoencoder.h"
#include "senna.h"

using namespace pugi;

namespace load_doc {

void load_file(TrainingCorpus& corpus, TrainingCorpus& doccorpus, string file_name,
     bool add_to_dict, Senna& senna, Senna& sennadoc);

Sentence createSentence(Senna& senna, vector<string>& words, int docsent_id,
                        bool add_to_dict=false);

}
#endif  // COMMON_LOAD_DOC_H
