// File: senna.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 08-02-2013
// Last Update: Fri 10 Jan 2014 05:02:57 PM GMT

// TODO(kmh): Note: This file is currently hacked to support subscripted
// embeddings. This really should be moved somewhere else (into the actual
// embeddings fiels for instance).

#include <iostream>
#include <fstream>

#include <Eigen/Core>

#include "shared_defs.h"

#include "senna.h"

#include "recursive_autoencoder.h"
#include "utils.h"

using namespace std;

  Senna::Senna(RecursiveAutoencoderBase& rae, int embeddings_type)
: good_counter(0), bad_counter(0), rae(rae)
{
  if (embeddings_type >= 0)
    use_embeddings = true;
  else
    use_embeddings = false;

  /*
   * Embedding types:
   * 0 - senna
   * 1 - turian
   * 2 - cldc en
   * 3 - cldc de
   */

  if (embeddings_type == 0)
  {
    string word_file = "senna/hash/words.lst";
    string embeddings_file = "senna/embeddings/embeddings.txt";
    string append = "_en";

    int count = 0;
    string line;
    {
      ifstream words_in(word_file.c_str());
      while (getline(words_in, line)) {
        line = trim(line) + append;
        words[trim(line)] = count;
        count++;
      }
    }

    Real token;
    count = 0;
    {
      ifstream embeddings_in(embeddings_file.c_str());
      while (getline(embeddings_in, line)) {
        vector<Real> instance;
        stringstream line_stream(line);
        while (line_stream >> token)
        {
          instance.push_back(token);
        }
        embeddings[count] = instance;
        count++;
      }
    }
  }
  else if (embeddings_type == 1)
  {
    // check for language!

    string embeddings_file = "../data/turian/x50.txt";
    string word;
    string line;
    Real token;
    int count = 0;
    string append = "_en";
    {
      ifstream embeddings_in(embeddings_file.c_str());
      while (getline(embeddings_in, line)) {
        vector<Real> instance;
        stringstream line_stream(line);
        line_stream >> word;
        word = trim(word) + append;
        words[trim(word)] = count;
        while (line_stream >> token)
        {
          instance.push_back(token);
        }
        embeddings[count] = instance;
        count++;
      }
    }
  }
  else if (embeddings_type > 2 && embeddings_type < 8)
  {
    string append = "";
    cout << "Reading in CLDC embeddings ";
    string embeddings_file = "";
    if (embeddings_type == 2) {
      cout << "in English (with German)" << endl;
      embeddings_file = "../data/alex/embeddings/de-en.de";
      append = "_en";
    }
    if (embeddings_type == 3) {
      cout << "in German" << endl;
      embeddings_file = "../data/alex/embeddings/de-en.en"; // don't ask me why - Alex mixed up the filenames it seems.
      append = "_de";
    }
    if (embeddings_type == 4) {
      cout << "in English (with French)" << endl;
      embeddings_file = "../data/alex/embeddings/fr-en.fr"; // don't ask me why - Alex mixed up the filenames it seems.
      append = "_en";
    }
    if (embeddings_type == 5) {
      cout << "in French" << endl;
      embeddings_file = "../data/alex/embeddings/fr-en.en"; // don't ask me why - Alex mixed up the filenames it seems.
      append = "_fr";
    }
    if (embeddings_type == 6) {
      cout << "in English (with Spanish)" << endl;
      embeddings_file = "../data/alex/embeddings/es-en.es"; // don't ask me why - Alex mixed up the filenames it seems.
      append = "_en";
    }
    if (embeddings_type == 7) {
      cout << "in Spanish" << endl;
      embeddings_file = "../data/alex/embeddings/es-en.en"; // don't ask me why - Alex mixed up the filenames it seems.
      append = "_es";
    }
    string word;
    string colon;
    string line;
    Real token;
    int count = 0;
    {
      ifstream embeddings_in(embeddings_file.c_str());
      while (getline(embeddings_in, line)) {
        vector<Real> instance;
        stringstream line_stream(line);
        line_stream >> word;
        line_stream >> colon;
        // line_stream >> colon;
        word = trim(word) + append;
        words[trim(word)] = count;
        while (line_stream >> token)
        {
          instance.push_back(token);
        }
        embeddings[count] = instance;
        count++;
      }
    }
  }
  else if (embeddings_type > 7)
  {
    string append = "";
    cout << "Reading in rmyeid embeddings ";
    string embeddings_file = "";
    if (embeddings_type ==  8) { embeddings_file = "../data/rmyeid/ar"; append = "_ar"; }
    if (embeddings_type ==  9) { embeddings_file = "../data/rmyeid/de"; append = "_de"; }
    if (embeddings_type == 10) { embeddings_file = "../data/rmyeid/en"; append = "_en"; }
    if (embeddings_type == 11) { embeddings_file = "../data/rmyeid/es"; append = "_es"; }
    if (embeddings_type == 12) { embeddings_file = "../data/rmyeid/fr"; append = "_fr"; }
    if (embeddings_type == 13) { embeddings_file = "../data/rmyeid/it"; append = "_it"; }
    if (embeddings_type == 14) { embeddings_file = "../data/rmyeid/nl"; append = "_nl"; }
    if (embeddings_type == 15) { embeddings_file = "../data/rmyeid/pb"; append = "_pb"; }
    if (embeddings_type == 16) { embeddings_file = "../data/rmyeid/pl"; append = "_pl"; }
    if (embeddings_type == 17) { embeddings_file = "../data/rmyeid/ro"; append = "_ro"; }
    if (embeddings_type == 18) { embeddings_file = "../data/rmyeid/ru"; append = "_ru"; }
    if (embeddings_type == 19) { embeddings_file = "../data/rmyeid/tr"; append = "_tr"; }
    if (embeddings_type == 20) { embeddings_file = "../data/rmyeid/zh"; append = "_zh"; }
    string word;
    string colon;
    string line;
    Real token;
    int count = 0;
    {
      ifstream embeddings_in(embeddings_file.c_str());
      while (getline(embeddings_in, line)) {
        vector<Real> instance;
        stringstream line_stream(line);
        line_stream >> word;
        // line_stream >> colon;
        // line_stream >> colon;
        word = trim(word) + append;
        words[trim(word)] = count;
        while (line_stream >> token)
        {
          instance.push_back(token);
        }
        embeddings[count] = instance;
        count++;
      }
    }
  }
  cout << "Read in " << words.size() << " words and " << embeddings.size() << " embeddings." << endl;
}


void Senna::applyEmbeddings()
{
  if (not use_embeddings)
    return;

  int found = 0;
  int notfound = 0;

  for (auto i = rae.dict_.min_label(); i<=rae.dict_.max_label(); ++i)
  {
    string dword = rae.dict_.label(i);
    auto j = words.find(dword);
    if (j != words.end())
    {
      found++;
      WeightVectorType x(&embeddings[j->second][0], embeddings[j->second].size());
      rae.D.row(i) = x;
    } else {
      notfound++;
    }
  }
  cout << "Found " << found << " of " << (found+notfound) << " words." << endl;

}

LabelID Senna::id(const Label& l, bool add_new)
{
  if ((not use_embeddings) or (not add_new)) {
    // return rae.dict_.id(l,add_new);
    LabelID x = rae.dict_.id(l,add_new);
    if (x != rae.dict_.m_bad_label) // Case 1: Word already in the dictionary
      ++good_counter;
    else {
      ++bad_counter;
    }
    return x;

  }
  else
  {
    // We use embeddings AND we want to add this word to the dictionary
    LabelID x = rae.dict_.id(l);
    if (x != rae.dict_.m_bad_label) // Case 1: Word already in the dictionary
      return x;
    else
    {
      auto i = words.find(l);
      if (i != words.end())           // Case 2: Word in embeddings
        return rae.dict_.id(l,true);  // add to dictionary
      else                            // Case 3: Word neither in dict nor in embeddings
        return rae.dict_.m_bad_label;
    }
  }
}
