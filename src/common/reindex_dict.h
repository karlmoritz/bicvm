// File: reindex_dict.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 14-02-2013
// Last Update: Fri 11 Oct 2013 03:53:14 PM BST
/*------------------------------------------------------------------------
 * Description: <DESC>
 *
 *------------------------------------------------------------------------
 * History:
 * TODO:
 *========================================================================

*/

#ifndef REINDEX_DICT_H_YD7SAOWK
#define REINDEX_DICT_H_YD7SAOWK

#include "dictionary.h"
#include "shared_defs.h"
//#include "sentsim.h"

class RecursiveAutoencoderBase;

RecursiveAutoencoderBase* reindex_dict(RecursiveAutoencoderBase& rae, TrainingCorpus& trainC);
RecursiveAutoencoderBase* reindex_dict(RecursiveAutoencoderBase& rae, TrainingCorpus& trainC, TrainingCorpus& testC);
//RecursiveAutoencoder reindex_dict(RecursiveAutoencoder& rae, SentSim::Corpus& trainC, SentSim::Corpus& testC);

#endif /* end of include guard: REINDEX_DICT_H_YD7SAOWK */
