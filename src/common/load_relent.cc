// File: load_relent.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 22-01-2013
// Last Update: Tue 20 May 2014 19:15:00 BST

// STL
#include <iostream>
#include <fstream>
#include <sstream>

#include "load_relent.h"
#include "dictionary.h"

using namespace std;

void load_relent::load_file(TrainingCorpus& corpus,
                        string file_name,
                        bool create_dict,
                        Senna& senna) {

  std::string line, type, word;
  int counter;
  std::ifstream file(file_name);
  while(std::getline(file, line)) {
    // Line format as follows:
    // <INT> <WORD>-<WORD>-<WORD>...<WORD>.<TYPE>
    // where <INT> is an integer referring to the entity in the database
    //       <WORD> are individual, lemmatised words, separated by hypens
    //       <TYPE> describes the type which is either 'e' or 'r'
    line = line.substr(0,line.length()-2); // remove type ending
    // cout << type << endl;
    // cout << line << endl;
    std::stringstream linestream(line);
    linestream >> counter; // discard integer.
    vector<string> words, wordsB;

    while(std::getline(linestream, word, '-')) {
      words.push_back(word);
    }

    if (int(words.size()) > 0) {
      corpus.push_back(createSentence(senna, words, create_dict));
    }
  }
}

Sentence load_relent::createSentence(Senna& senna, vector<string>& words, bool add_to_dict) {
  Sentence instance(0);
  for(int word_count = 0; word_count < int(words.size()); ++word_count) {
    instance.words.push_back(senna.id(words[word_count], add_to_dict));
  }
  return instance;
}
