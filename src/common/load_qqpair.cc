// File: load_qqpair.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 22-01-2013
// Last Update: Mon 26 May 2014 14:46:15 BST

// STL
#include <iostream>
#include <fstream>
#include <sstream>

#include "load_qqpair.h"
#include "dictionary.h"

using namespace std;

void load_qqpair::load_file(TrainingCorpus& corpus_string,
                            TrainingCorpus& corpus_query1,
                            TrainingCorpus& corpus_query2,
                            string file_name,
                            bool create_dict,
                            bool remove_type_ending,
                            Senna& senna) {

  std::string line, partline, compound, word;
  int truth_label;
  std::ifstream file(file_name);
  while(std::getline(file, line)) {
    // Line format as follows:
    // <INT>\t<QUESTION>\t<REL>.r <ENT1>.e <ENT2>.e
    std::stringstream linestream(line);
    vector<string> wordsA, wordsB, wordsC;

    // Truth label (0 or 1)
    {
      std::getline(linestream, partline, '\t');
      std::stringstream ss(partline);
      ss >> truth_label;
    }

    // Question sentence.
    {
      std::getline(linestream, partline, '\t');
      std::stringstream ss(partline);
      while (ss >> word) { // reading in the first sentence
        wordsA.push_back(word);
      }
    }

    // Query sentence.
    {
      std::getline(linestream, partline, '\t');
      std::stringstream ss(partline);

      {
      ss >> compound; // relation
      compound = compound.substr(0,compound.length()-2); // remove type ending
      std::stringstream compoundstream(compound);
      while(std::getline(compoundstream, word, '-')) {
        wordsB.push_back(word);
        wordsC.push_back(word);
      }
      }

      {
        ss >> compound; // entity 1
        if ( remove_type_ending ) {
          compound = compound.substr(0,compound.length()-2); // remove type ending
          std::stringstream compoundstream(compound);
          while(std::getline(compoundstream, word, '-')) {
            wordsB.push_back(word);
          }
        } else {
          wordsB.push_back(compound + ".1");
        }
      }

      {
        ss >> compound; // entity 2
        if ( remove_type_ending ) {
          compound = compound.substr(0,compound.length()-2); // remove type ending
          std::stringstream compoundstream(compound);
          while(std::getline(compoundstream, word, '-')) {
            wordsC.push_back(word);
          }
        } else {
          wordsC.push_back(compound + ".2");
        }
      }
    }

    if (int(wordsA.size()) > 0) {
      corpus_string.push_back(createSentence(senna, wordsA, truth_label));
      corpus_query1.push_back(createSentence(senna, wordsB));
      corpus_query2.push_back(createSentence(senna, wordsC));
    }
  }
}

Sentence load_qqpair::createSentence(Senna& senna, vector<string>& words, int truth_label) {
  Sentence instance(truth_label);
  for(int word_count = 0; word_count < int(words.size()); ++word_count) {
    instance.words.push_back(senna.id(words[word_count], false));
  }
  return instance;
}
