// File: load_plain.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 22-01-2013
// Last Update: Thu 31 Oct 2013 05:27:29 PM GMT

// STL
#include <iostream>
#include <fstream>
#include <sstream>
#include <queue>

#include "load_plain.h"
#include "dictionary.h"

using namespace pugi;
using namespace std;

struct qelem
{
  xml_node node;
  int id;
  qelem(xml_node n, int i) : node(n), id(i) {}
};

typedef queue<qelem> xqueue;

void load_plain::load(TrainingCorpus& trainCorpus, TrainingCorpus& testCorpus,
    string file_positive, string file_negative, bool add_to_dict, Senna& senna)
{

  load_file(trainCorpus,file_positive,0,add_to_dict,senna);
  if (!file_negative.empty()) // rae only training doesn't have a negative file ...
    load_file(trainCorpus,file_negative,1,add_to_dict,senna);
  testCorpus = trainCorpus;

  cout << "Test Set Size " << testCorpus.size() << endl;
  cout << "Train Set Size " << trainCorpus.size() << endl;


}

void load_plain::load_file(TrainingCorpus& corpus, string file_name, int label,
                           bool add_to_dict, Senna& senna) {
  string line;
  std::ifstream file(file_name);
  while(std::getline(file, line)) {
    std::stringstream ss(line);
    std::string  word;
    vector<string> words;
    while (ss >> word) {
      words.push_back(word);
    }
    if (int(words.size()) > 0) {
      corpus.push_back(createSentence(senna, words, label, add_to_dict));
    }
  }
}

Sentence load_plain::createSentence(Senna& senna, vector<string>& words, int label, bool add_to_dict) {
  Sentence instance(label);
  for(int word_count = 0; word_count < int(words.size()); ++word_count) {
    instance.words.push_back(senna.id(words[word_count], add_to_dict));
  }
  return instance;
}
