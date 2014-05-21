// File: utils.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 30-01-2013
// Last Update: Wed 21 May 2014 19:42:25 BST

#ifndef COMMON_UTILS_H
#define COMMON_UTILS_H

#include <algorithm>
#include <functional>
#include <cctype>
#include <locale>

#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>

#include "shared_defs.h"
#include "models.h"

namespace bpo = boost::program_options;

void dumpModel(Model& model, int k);
void printSentence(const Dictionary& dict, const Sentence &sent);
void paraphraseTest(Model& model, int k);
void printConfig(const bpo::variables_map& vm);

// trim from start
static inline std::string &ltrim(std::string &s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
        return s;
}

// trim from end
static inline std::string &rtrim(std::string &s) {
        s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
        return s;
}

// trim from both ends
static inline std::string &trim(std::string &s) {
        return ltrim(rtrim(s));
}

#endif  // COMMON_UTILS_H
