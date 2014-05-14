// File: load_qq.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 22-01-2013
// Last Update: Wed 14 May 2014 16:01:51 BST

// STL
#include <iostream>
#include <fstream>
#include <sstream>
#include <queue>

#include "load_qq.h"
#include "dictionary.h"

using namespace pugi;
using namespace std;

struct qelem {
  xml_node node;
  int id;
  qelem(xml_node n, int i) : node(n), id(i) {}
};

typedef queue<qelem> xqueue;

void load_qq::load_file(TrainingCorpus& corpusA,
                        TrainingCorpus& corpusB,
                        string file_name,
                        bool create_dict_A,
                        bool create_dict_B,
                        Senna& sennaA,
                        Senna& sennaB) {

  string line;
  std::ifstream file(file_name);
  while(std::getline(file, line)) {
    std::stringstream ss(line);
    std::string  word;
    vector<string> words, query;
    while (ss >> word) { // reading in the question
      if (word == "(lambda") break;
      words.push_back(word);
    }
    while (ss >> word) { // reading in the question
      query.push_back(word);
    }
    if (int(words.size()) > 0) {
      corpusA.push_back(createSentence(sennaA, words, create_dict_A));
      corpusB.push_back(createSentence(sennaB, query, create_dict_B));
    }
  }
}

Sentence load_qq::createSentence(Senna& senna, vector<string>& words, bool add_to_dict) {
  Sentence instance(0);
  for(int word_count = 0; word_count < int(words.size()); ++word_count) {
    instance.words.push_back(senna.id(words[word_count], add_to_dict));
  }
  return instance;
}
