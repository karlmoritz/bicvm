// File: dictionary.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 31-12-2012
// Last Update: Fri 03 Jan 2014 05:01:11 PM GMT
/*------------------------------------------------------------------------
 * Description: <DESC>
 *
 *------------------------------------------------------------------------
 * History:
 * TODO:
 *========================================================================
 */

#ifndef __DICTIONARY_H__
#     define __DICTIONARY_H__

#include <boost/bimap.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <iostream>

typedef std::string     Label;
typedef int             LabelID;
typedef boost::bimap<Label,LabelID>   LabelBimap;
typedef LabelBimap::left_value_type   LabelPair;

using namespace std;

class Dictionary
{
public:

  Dictionary() : m_bad_label(int(0)), m_bad_id("_UNK_") {
    int x = id(m_bad_id,true);
    assert(x == m_bad_label);
  }

  Dictionary(int bad) : m_bad_label(bad), m_bad_id("_UNK_") {
  }

  LabelID id(const Label& l) const {
    auto i = m_bimap.left.find(l);
    return (i==m_bimap.left.end() ? m_bad_label : i->second);
  }

  LabelID id(const Label& l, bool add_new=false) {
    if (add_new) {
      auto i = m_bimap.left.insert(LabelPair(l,int(m_bimap.size())));
      return i.first->second;
    }
    else {
      auto i = m_bimap.left.find(l);
      return (i==m_bimap.left.end() ? m_bad_label : i->second);
    }
  }

  Label label(const LabelID& l) const {
    auto i = m_bimap.right.find(l);
    return (i==m_bimap.right.end() ? m_bad_id : i->second);
  }

  LabelID min_label() const
  { return 0; }

  LabelID max_label() const
  { return int(m_bimap.size()) - 1; }

  LabelID num_labels() const
  { return max_label()-min_label() + 1; }

  bool valid(const LabelID& l) const
  { return l >= min_label() && l <= max_label(); }

  friend class boost::serialization::access;
  template<class Archive>
    void save(Archive& ar, const unsigned version) const {
      ar & const_cast<const LabelBimap&>(m_bimap);
    }

  template<class Archive>
    void load(Archive& ar, const unsigned version) {
      ar & m_bimap;
    }
  BOOST_SERIALIZATION_SPLIT_MEMBER()

  const LabelID   m_bad_label;
  const Label     m_bad_id;

protected:
  LabelBimap      m_bimap;
};

#endif /* __DICTIONARY_H__ */
