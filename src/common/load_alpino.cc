// File: load_alpino.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 22-01-2013
// Last Update: Thu 17 Oct 2013 04:08:03 PM BST

// STL
#include <iostream>
#include <fstream>
#include <queue>

#include "load_alpino.h"

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

void load_alpino::load(TrainingCorpus& trainCorpus, TrainingCorpus& testCorpus,
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

void load_alpino::load_file(TrainingCorpus& corpus, string file_name,
                            int cv_split, bool use_cv, bool load_test,
                            int label, bool add_to_dict, Senna& senna) {
  int counter = 0;
  string line, token;

  pugi::xml_document doc;
  if (!doc.load_file(file_name.c_str())) assert(false);
  xml_node root = doc.child("treebank");
  for (xml_node sentence = root.first_child(); sentence; sentence = sentence.next_sibling())
  {
    if (use_cv)
    {
      if (load_test && counter%10==cv_split)
      {
        Sentence instance(label);
        createSentence(instance, sentence, add_to_dict, senna);
        corpus.push_back(instance);
      }
      else if (not load_test && counter%10 != cv_split)
      {
        Sentence instance(label);
        createSentence(instance, sentence, add_to_dict, senna);
        corpus.push_back(instance);
      }
    }
    else
    {
      Sentence instance(label);
      createSentence(instance, sentence, add_to_dict, senna);
      corpus.push_back(instance);
    }
    counter++;
  }
}

void load_alpino::createSentence(Sentence& instance, xml_node& sentence,
                                    bool add_to_dict, Senna& senna)
{

  xqueue q;

  // Variables
  std::string parent_name;
  std::string field_;
  int cat_;
  CCGRuleType rule_;

  int counter = 0;
  xml_node active_child;
  q.push(qelem(sentence.first_child(),counter));
  counter ++;

  while (not q.empty())
  {
    qelem parent = q.front();
    q.pop();
    parent_name = parent.node.name();
    instance.child0.push_back(-1);
    instance.child1.push_back(-1);

    if(parent_name == "node" and parent.node.attribute("cat").value())
    {
      // All will be other (no POS tag list for now)
      field_ = parent.node.attribute("cat").value();
      rule_ = (s2r_map.find( field_ ) != s2r_map.end()) ? s2r_map[field_] : OTHER;
      //field_ = parent.node.attribute("cat").value();
      cat_ = (c2r_map.find( field_ ) != c2r_map.end()) ? c2r_map[field_] : 0;
      //cout << "Cat " << field_ << ": " << cat_ << endl;
      assert(rule_ != LEAF);

      instance.nodes.push_back(-1);
      instance.rule.push_back(rule_);
      instance.cat.push_back(cat_);
      //cout << "Rule: " << field_ << " " << rule_ << endl;
    }
    else if(parent_name == "node" and parent.node.attribute("word").value())
    {
      //field_ = parent.node.attribute("cat").value();
      cat_ = 0; //(c2r_map.find( field_ ) != c2r_map.end()) ? c2r_map[field_] : 0;
      field_ = parent.node.attribute("word").value();
      instance.words.push_back(senna.id(field_,add_to_dict));
      instance.nodes.push_back(int(instance.words.size())-1);
      instance.rule.push_back(LEAF);
      instance.cat.push_back(cat_);
    }
    else
    {
      cout << "PN" << parent_name << endl;
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
}
