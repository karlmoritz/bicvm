bicvm
===

BiCVM Code


Dependencies
====

This code require several libraries in order to run:

* PugiXML (included)
* libSVM (http://www.chokkan.org/software/liblbfgs/)
* Boost
* Eigen


To Do
====

This is development code and not fully functional. Currently, dbltrain should
behave as expected; the other executables won't compile and still need to be
ported to the refactored codebase.

Dictionary/Model initialisation could be cleaned up!

More ToDo: There is some technical debt from the memory optimisation on the
corpus side. Principally this concerns the push_back overload for the corpus
function - this needs to be completed given more complex models.
