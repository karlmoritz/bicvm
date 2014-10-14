// File: reindex_dict.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 14-02-2013
// Last Update: Mon 15 Sep 2014 15:22:56 BST

#include "reindex_dict.h"
#include "recursive_autoencoder.h"


/***************************************************************************
 *           Take an de and reindex it based on a given corpus            *
 ***************************************************************************/
DictionaryEmbeddings* reindex_dict(DictionaryEmbeddings& de, TrainingCorpus& trainC)
{
  DictionaryEmbeddings* new_de = new DictionaryEmbeddings(de.getWordWidth());
  std::map<LabelID,LabelID> n2o_map;

  // A: Populate new dictionary based on old one
  // for (auto instance = trainC.begin(); instance != trainC.end(); ++instance)
    // for (auto word = instance->words.begin(); word != instance->words.end(); ++word)
  for (int i = 0; i < trainC.words.size(); ++i)
    for(int j = 0; j < trainC.words[i].size(); ++j)
    {
      n2o_map[new_de->dict_.id(de.dict_.label(trainC[i][j]),true)] = trainC[i][j];
      trainC[i][j] = new_de->dict_.id(de.dict_.label(trainC[i][j]),true);
    }
  // Allocate space for new dictionary. ZERO embedding for unknown words.
  // (TODO: Think whether this should be random?).
  new_de->init(false,true);
  // Copy embeddings from old dictionary.
  new_de->initFromDict(de,n2o_map);
  cout << "Reindexed dictionary from " << de.getDictSize() << " entries down to " << new_de->getDictSize() << "." << endl;
  return new_de;
}

DictionaryEmbeddings* reindex_dict(DictionaryEmbeddings& de, TrainingCorpus& trainC, TrainingCorpus& testC)
{
  DictionaryEmbeddings* new_de = new DictionaryEmbeddings(de.getWordWidth());
  std::map<LabelID,LabelID> n2o_map;

  // A: Populate new dictionary based on old one
  // for (auto instance = trainC.begin(); instance != trainC.end(); ++instance)
    // for (auto word = instance->words.begin(); word != instance->words.end(); ++word)
    // {
      // n2o_map[new_de->dict_.id(de.dict_.label(*word),true)] = *word;
      // *word = new_de->dict_.id(de.dict_.label(*word),true);
  for (int i = 0; i < trainC.words.size(); ++i)
    for(int j = 0; j < trainC.words[j].size(); ++j)
    {
      n2o_map[new_de->dict_.id(de.dict_.label(trainC[i][j]),true)] = trainC[i][j];
      trainC[i][j] = new_de->dict_.id(de.dict_.label(trainC[i][j]),true);
    }

  if (&testC != &trainC)
    for (int i = 0; i < testC.words.size(); ++i)
      for(int j = 0; j < testC.words[j].size(); ++j)
      {
        n2o_map[new_de->dict_.id(de.dict_.label(testC[i][j]),true)] = testC[i][j];
        testC[i][j] = new_de->dict_.id(de.dict_.label(testC[i][j]),true);
      }

  // Allocate space for new dictionary. ZERO embedding for unknown words.
  // (TODO: Think whether this should be random?).
  new_de->init(false,true);
  // Copy embeddings from old dictionary.
  new_de->initFromDict(de,n2o_map);
  cout << "Reindexed dictionary from " << de.getDictSize() << " entries down to " << new_de->getDictSize() << "." << endl;
  return new_de;
}
