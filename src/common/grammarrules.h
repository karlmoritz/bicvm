// File: grammarrules.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 11-02-2013
// Last Update: Thu 17 Oct 2013 02:52:07 PM BST
/*------------------------------------------------------------------------
 * Description: <DESC>
 *
 *------------------------------------------------------------------------
 * History:
 * TODO:
 *========================================================================
 */

#ifndef TAGDEF_H_LZJBNV4H
#define TAGDEF_H_LZJBNV4H


// STL
#include <iostream>
#include <boost/assign.hpp>
#include <string>

using namespace std;

static int num_ccg_rules = 20;

enum CCGRuleType {
  LEAF = -1,   // Special type for leaf nodes
  OTHER = 0,   // Catch all. Set to 1 as Wx[0] is the generalized matrix
  FA,          // 60k
  BA,          // 25k
  LEX,         // 14k
  CONJ,        // 6k
  RP,          // 8k
  LP,          // 2k
  BX,          // 1k
  TR,          // 1k
  FC,          // 1k
  BC,          // 300
  FUNNY,       // 160
  RTC,         // 110
  LTC,         // 100
  GBX,         // 14
  GBC,         // 0
  GFC,         // 0
  GOTHER,      // 0
  APPO,        // 0
  _UNK_,
};

static std::map<string,CCGRuleType> s2r_map =
boost::assign::map_list_of
("leaf",LEAF)
("other",OTHER)
("fa",FA)
("ba",BA)
("lex",LEX)
("conj",CONJ)
("rp",RP)
("lp",LP)
("bx",BX)
("tr",TR)
("fc",FC)
("bc",BC)
("funny",FUNNY)
("rtc",RTC)
("ltc",LTC)
("gbx",GBX)
("gbc",GBC)
("gfc",GFC)
("gother",GOTHER)
("appo",APPO)
("_UNK_",_UNK_);


static int num_ccg_types = 26;

static std::map<string,int> c2r_map =
boost::assign::map_list_of
("OTHER",0)
("N",1) // Count: 516489
("NP",2)        // Count: 390344
("S[dcl]",3)    // Count: 209220
("N/N",4)       // Count: 177776
("NP[nb]",5)    // Count: 169054
("S[dcl]\\NP",6)        // Count: 162391
("NP[nb]/N",7)  // Count: 161325
("NP\\NP",8)    // Count: 151296
("(NP\\NP)/NP",9)       // Count: 75891
(",",10) // Count: 70680
("(S[X]\\NP)\\(S[X]\\NP)",11)   // Count: 70562
(".",12)        // Count: 69379
("S[b]\\NP",13) // Count: 68185
("conj",14)     // Count: 45861
("(S[dcl]\\NP)/NP",15)  // Count: 45444
("S[pss]\\NP",16)       // Count: 41865
("((S\\NP)\\(S\\NP))/NP",17)    // Count: 37978
("S[adj]\\NP",18)       // Count: 36821
("(S\\NP)\\(S\\NP)",19) // Count: 34632
("PP",20)       // Count: 31215
("S[ng]\\NP",21)        // Count: 30346
("PP/NP",22)    // Count: 29738
("(S[dcl]\\NP)/(S[b]\\NP)",23)  // Count: 27309
("(S[b]\\NP)/NP",24)    // Count: 27070
("(S[to]\\NP)/(S[b]\\NP)",25);   // Count: 22955

/*
 *("S[to]\\NP",26)        // Count: 22617
 *("S[X]/S[X]",27)        // Count: 19736
 *("S[dcl]\\S[dcl]",28)   // Count: 14692
 *("S[pt]\\NP",29)        // Count: 14570
 *("LQU",30)      // Count: 12591
 *("(S[dcl]\\NP)/(S[adj]\\NP)",31)        // Count: 12259
 *("RQU",32)      // Count: 12184
 *("N\\N",33)     // Count: 11218
 *("(S[ng]\\NP)/NP",34)   // Count: 11211
 *("(S[dcl]\\NP)/(S[pt]\\NP)",35) // Count: 10679
 *("(S[dcl]\\NP)/(S[pss]\\NP)",36)        // Count: 10551
 *;
 */

static int number_stf_types = 46;


static std::map<string,int> p2r_map =
boost::assign::map_list_of
("LEAF",-1)
("OTHER",0)
("AA",1) // kill
("ADJP",2)
("ADVP",3)
("AP",4)
("AVP",5)
("CAC",6) // kill
("CAP",7)
("CAVP",8) // kill
("CCP",9)  // kill
("CH",10) // kill
("CNP",11)
("CO",12)    //  kill
("CONJP",13)// kill
("CPP",14)     //  kill
("CS",15)
("CVP",16)
("CVZ",17) // kill
("DL",18)
("FRAG",19)
("INTJ",20)
("ISU",21)
("LST",22)
("MPN",23)
("MTA",24)
("NAC",25)
("NM",26)
("NP",27)
("NX",28)
("PP",29)
("PRN",30)
("QL",31)
("QP",32)
("RRC",33)
("S",34)
("SBAR",35)
("SBARQ",36)
("SINV",37)
("SQ",38)
("UCP",39)
("VP",40)
("VZ",41)
("WHADJP",42)
("WHAVP",43)
("WHNP",44)
("_UNK_",45);


#endif /* end of include guard: TAGDEF_H_LZJBNV4H */
