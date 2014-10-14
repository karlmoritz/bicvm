// File: load_doc.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 22-01-2013
// Last Update: Mon 06 Jan 2014 05:55:23 PM GMT

// STL
#include <iostream>
#include <fstream>
#include <sstream>
#include <queue>

// Boost
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/filesystem/fstream.hpp>

#include "load_doc.h"
#include "dictionary.h"

using namespace pugi;
using namespace std;
namespace fs = boost::filesystem;

struct qelem
{
  xml_node node;
  int id;
  qelem(xml_node n, int i) : node(n), id(i) {}
};

typedef queue<qelem> xqueue;

void load_doc::load_file(TrainingCorpus& corpus, TrainingCorpus& doccorpus,
                         string input_dir, bool add_to_dict, Senna& senna,
                         Senna& sennadoc) {

  std::string  line, word, docsent;
  int docsent_id, sentence_count, doc_count = 0;
  fs::path sourceDir(input_dir);
  fs::directory_iterator it(sourceDir), eod;
  BOOST_FOREACH(fs::path const &p, std::make_pair(it, eod))
  {
    // Iterate over every file in the directory. We assume matching file names
    // across the two language directories, unique across multiple language
    // directories and use these as document identifiers.
    if(is_regular_file(p))
    {
      fs::ifstream file(p);
      Sentence docsentence(0);
      sentence_count = 0;
      while(std::getline(file, line)) {
        if (line[0] == '<') { continue; }
        std::stringstream ss(line);
        vector<string> words;
        while (ss >> word) {
          words.push_back(word);
        }
        if (int(words.size()) > 0) {
          docsent = p.string() + boost::lexical_cast<std::string>(sentence_count);
          docsent_id = sennadoc.id(docsent, true);
          docsentence.words.push_back(docsent_id);
          corpus.push_back(createSentence(senna, words, docsent_id, add_to_dict));
          ++sentence_count;
        }
      }
      if (int(docsentence.words.size()) > 0) {
        docsentence.id = doc_count;
        if (&senna != &sennadoc) // only do this thing if we have two corpora.
          doccorpus.push_back(docsentence);
      }
      ++doc_count;
    }
  }
}

Sentence load_doc::createSentence(Senna& senna, vector<string>& words,
                                  int docsent_id, bool add_to_dict) {
  Sentence instance(0);
  instance.id = docsent_id;
  for(int word_count = 0; word_count < int(words.size()); ++word_count) {
    instance.words.push_back(senna.id(words[word_count], add_to_dict));
  }
  return instance;
}
