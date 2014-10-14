// File: load_stanford.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 22-01-2013
// Last Update: Thu 17 Oct 2013 04:31:01 PM BST
/*------------------------------------------------------------------------
 * Description: <DESC>
 *
 *------------------------------------------------------------------------
 * History:
 * TODO:
 *========================================================================
 */


// STL
#include <iostream>
#include <fstream>
#include <queue>
#include <algorithm>
#include <string>

# if defined(RECENT_BOOST)
#include <boost/locale.hpp>
# endif

#include "load_stanford.h"

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

void load_stanford::load(TrainingCorpus& trainCorpus, TrainingCorpus& testCorpus,
    string file_positive, string file_negative,
    int cv_split, bool use_CV, bool add_to_dict, Senna& senna)
{

  if (use_CV)
  {
    load_file(trainCorpus,file_positive,cv_split,true,false,0,add_to_dict,senna);
    if (!file_negative.empty()) // rae only training doesn't have a negative file ...
      load_file(trainCorpus,file_negative,cv_split,true,false,1,add_to_dict,senna);
    load_file(testCorpus,file_positive,cv_split,true,true,0,false,senna);
    if (!file_negative.empty()) // rae only training doesn't have a negative file ...
      load_file(testCorpus,file_negative,cv_split,true,true,1,false,senna);
  }
  else
  {
    load_file(trainCorpus,file_positive,-1,false,false,0,add_to_dict,senna);
    if (!file_negative.empty()) // rae only training doesn't have a negative file ...
      load_file(trainCorpus,file_negative,-1,false,false,1,add_to_dict,senna);
    testCorpus = trainCorpus;
  }

  cout << "Test Set Size " << testCorpus.size() << endl;
  cout << "Train Set Size " << trainCorpus.size() << endl;


}

void load_stanford::load_file(TrainingCorpus& corpus, string file_name,
     int cv_split, bool use_cv, bool load_test, int label, bool add_to_dict, Senna& senna)
{
  int counter = 0;
  string line, token;

  pugi::xml_document doc;
  if (!doc.load_file(file_name.c_str())) assert(false);
  xml_node root = doc.child("PARSE");
  for (xml_node sentence = root.first_child(); sentence; sentence = sentence.next_sibling())
  {
    std::string n = sentence.name();
    if (n != "ROOT") // assume parse error (empty sentence) - skip it
      continue;
    if (use_cv)
    {
      if (load_test && counter%10==cv_split)
      {
        Sentence instance(label);
        createSentence(instance, sentence,  add_to_dict, senna);
        corpus.push_back(instance);
      }
      else if (not load_test && counter%10 != cv_split)
      {
        Sentence instance(label);
        createSentence(instance, sentence,  add_to_dict, senna);
        corpus.push_back(instance);
      }
    }
    else
    {
      Sentence instance(label);
      createSentence(instance, sentence,  add_to_dict, senna);
      corpus.push_back(instance);
    }
    counter++;
  }
}

void load_stanford::createSentence(Sentence& instance, xml_node& sentence,   bool add_to_dict, Senna& senna)
{

  xqueue q;

  // Variables
  std::string parent_name;
  std::string field_;
  int cat_;

  int counter = 0;
  xml_node active_child;
  if (sentence.first_child()) {
    q.push(qelem(sentence.first_child(),counter));
  } else {
    cout << "Empty sentence (parsing error?)" <<endl;
    assert(false);
  }
  counter ++;

  while (not q.empty())
  {
    qelem parent = q.front();
    q.pop();
    parent_name = parent.node.name();
    instance.child0.push_back(-1);
    instance.child1.push_back(-1);

    if(parent_name == "pos")
    {
      field_ = parent.node.attribute("tag").value();
      cat_ = (p2r_map.find( field_ ) != p2r_map.end()) ? p2r_map[field_] : 0;
      instance.nodes.push_back(-1);
      instance.rule.push_back(OTHER);
      instance.cat.push_back(cat_);
      instance.tree_size.push_back(0);

    }
    else if(parent_name == "elem")
    {
      cat_ = -1; //(c2r_map.find( field_ ) != c2r_map.end()) ? c2r_map[field_] : 0;
      field_ = parent.node.attribute("word").value();

# if defined(RECENT_BOOST)
      field_ = boost::locale::to_lower(field_); // need boost lower thingy because of German umlauts
# endif
      //cout << field_ << endl;
      instance.words.push_back(senna.id(field_,add_to_dict));
      instance.nodes.push_back(int(instance.words.size())-1);
      instance.rule.push_back(LEAF);
      instance.cat.push_back(cat_);
      instance.tree_size.push_back(1);
    }
    else
    {
      cout << "PN" << parent_name << endl;
      for (pugi::xml_attribute_iterator ait = parent.node.attributes_begin(); ait != parent.node.attributes_end(); ++ait)
      {
        std::cout << " " << ait->name() << "=" << ait->value();
      }
      //continue;
      assert(false);
    }

    xml_node active_child = parent.node.first_child();
    if (active_child)
    {
      q.push(qelem(active_child,counter));
      instance.child0[parent.id] = counter;
      counter++;

      active_child = active_child.next_sibling();
      if (active_child)
      {
        q.push(qelem(active_child,counter));
        instance.child1[parent.id] = counter;
        counter++;

        if (active_child.next_sibling())
        {
          cout << "triple" << endl;
          cout << active_child.attribute("word").value();
          cout << active_child.attribute("tag").value();

        }
        assert (not active_child.next_sibling());
      }
    }
  }
  // Fix tree counts
  for (int i = int(instance.rule.size()) - 1; i >= 0; --i) {
    if(instance.rule[i] != LEAF) {
      if(instance.child0[i] != -1)
        instance.tree_size[i] += instance.tree_size[ instance.child0[i] ];
      if(instance.child1[i] != -1)
        instance.tree_size[i] += instance.tree_size[ instance.child1[i] ];
    }
  }
}
