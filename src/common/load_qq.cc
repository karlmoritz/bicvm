// File: load_qq.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 22-01-2013
// Last Update: Tue 20 May 2014 19:02:24 BST

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

  std::string line;
  std::string partline;
  std::string word;
  std::ifstream file(file_name);
  while(std::getline(file, line)) {
    // Line separated by three tabs. Sentence 1 TAB Sentence 2 TAB Alignments
    // We want sentence one and sentence two and discard the alignments.
    std::stringstream linestream(line);
    vector<string> wordsA, wordsB;

    // First sentence
    {
      std::getline(linestream, partline, '\t');
      std::stringstream ss(partline);
      while (ss >> word) { // reading in the first sentence
        wordsA.push_back(word);
      }
    }
    // Second sentence
    {
      std::getline(linestream, partline, '\t');
      std::stringstream ss(partline);
      while (ss >> word) { // reading in the second sentence
        wordsB.push_back(word);
      }
    }

    if (int(wordsA.size()) > 0) {
      corpusA.push_back(createSentence(sennaA, wordsA, create_dict_A));
      corpusB.push_back(createSentence(sennaB, wordsB, create_dict_B));
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
