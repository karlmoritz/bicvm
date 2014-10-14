// File: load_alpino.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 22-01-2013
// Last Update: Thu 17 Oct 2013 04:04:06 PM BST
/*------------------------------------------------------------------------
 * Description: <DESC>
 *
 *------------------------------------------------------------------------
 * History:
 * TODO:
 *========================================================================
 */

#ifndef COMMON_LOAD_ALPINO_H
#define COMMON_LOAD_ALPINO_H

#include "../pugi/pugixml.hpp"

#include "shared_defs.h"
#include "recursive_autoencoder.h"
#include "senna.h"

using namespace pugi;

namespace load_alpino {
void load(TrainingCorpus& trainCorpus, TrainingCorpus& testCorpus, string
    file_positive, string file_negative, int
    cv_split, bool use_CV, bool add_to_dict, Senna& senna);

void load_file(TrainingCorpus& corpus, string file_name,
    int cv_split, bool use_cv, bool load_test, int label, bool add_to_dict,
    Senna& senna);

void createSentence(Sentence& instance, xml_node& sentence,
    bool add_to_dict, Senna& senna);
}

#endif  // COMMON_LOAD_ALPINO_H
