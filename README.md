BiCVM
======

BiCVM code for learning distributed representations in a variety of settings. In
particular, this code can be used to reproduce the results of the paper
"Multilingual Models for Compositional Distributional Semantics" (Hermann and
Blunsom, ACL 2104).

Since publishing this paper, the code-base has been rewritten quite
significantly to allow training on significantly larger data than was previously
possible.

Dependencies
====

This code require several libraries in order to run:

* PugiXML (included)
* libLBFGS (http://www.chokkan.org/software/liblbfgs/)
* Boost
* Eigen

Installation
====

Install via CMake:
```
   mkdir Release
   cd Release
   cmake ../src
   make
```

Usage
====

* **dbltrain** - Learns embeddings given two files of parallel, sentence aligned
text
* **doctrain** - Learns embeddings given two folders of sentence aligned
documents (matching names in folders)

More documentation to come - please contribute!

One simple example
```
./dbltrain --input1 sample/english \
    --input2 sample/german \
    --tree plain \
    --type additive \
    --method adagrad \
    --word-width 128 \
    --hinge_loss_margin 128 \
    --model1-out modelA.en.128 \
    --model2-out modelB.de.128 \
    --noise 10 \
    --batches 10 \
    --eta 0.05 \
    --lambdaD 1 \
    --calc_bi_error1 true \
    --calc_bi_error2 true \
    --iterations 5
```
should result in the following output:
```
BiCVM Distributed Representation Learner: Copyright 2013-2014 Karl Moritz Hermann
################################
# Config Summary
# alpha = 0.2
# batches = 10
# calc_bi_error1 = 1
# calc_bi_error2 = 1
# calc_lbl_error1 = 0
# calc_lbl_error2 = 0
# calc_rae_error1 = 0
# calc_rae_error2 = 0
# calc_thr_error1 = 0
# calc_thr_error2 = 0
# calc_uae_error1 = 0
# calc_uae_error2 = 0
# cv-split = -1
# dump-frequency = 10
# dynamic-mode = 0
# embeddings = -1
# epsilon = 1e-06
# eta = 0.05
# ftcbatches = 100
# ftceta = 0.01
# ftiterations = 1000
# gamma = 0.1
# hinge_loss_margin = 128
# initI = 1
# input1 = sample/english
# input2 = sample/german
# iterations = 9
# l1 = 0
# lambdaBd = 1
# lambdaBl = 1
# lambdaD = 1
# lambdaWd = 1
# lambdaWl = 1
# linesearch = armijo
# method = adagrad
# model1-out = modelA.en.128
# model2-out = modelB.de.128
# noise = 10
# norm = 0
# num-sentences = 0
# tree = plain
# type = additive
# updateD1 = 1
# updateD2 = 1
# updateF1 = 1
# updateF2 = 1
# updateWd1 = 1
# updateWd2 = 1
# updateWl1 = 1
# updateWl2 = 1
# word-width = 128
# History
#  | adagrad(armijo) it:9 lambdas: /1/
################################
Read in 0 words and 0 embeddings.
Read in 0 words and 0 embeddings.
L1 Size 10
L2 Size 10
Reindexed dictionary from 19 entries down to 19.
Reindexed dictionary from 22 entries down to 22.
Dict size: 19 and 22
Training with AdaGrad
Batch size: 2  eta 0.05
Iteration 0
Error	524.989
Iteration 1
Error	483.947
Iteration 2
Error	434.471
Iteration 3
Error	367.271
Iteration 4
Error	281.668
Iteration 5
Error	179.812
Iteration 6
Error	86.0947
Iteration 7
Error	27.2102
Iteration 8
Error	5.8242
```

Until there is more documentation, please refer to the code and the "--help"
commands for more information on usage.

References
====

If you use this software package in your experiments and publish related work,
   please cite one of the following papers as appropriately:

For most work, the following paper should be cited, in which this code was
introduced, as well ash the recursive document-level model.
```
@InProceedings{Hermann:2014:ACLphil,
  author    = {Hermann, Karl Moritz and Blunsom, Phil},
  title     = {{Multilingual Models for Compositional Distributional Semantics}},
  booktitle = {Proceedings of ACL},
  year      = {2014},
  month     = jun,
  url       = {http://arxiv.org/abs/1404.4641},
}
```

For the basic noise-contrastive/large margin objective function over parallel
data:

```
@InProceedings{Hermann:2014:ICLR,
  author    = {Hermann, Karl Moritz and Blunsom, Phil},
  title     = {{Multilingual Distributed Representations without Word Alignment}},
  booktitle = {Proceedings of ICLR},
  year      = {2014},
  month     = apr,
  url       = {http://arxiv.org/abs/1312.6173},
}
```

For anything related to syntax-based composition models:
```
@InProceedings{Hermann:2013:ACL,
  author    = {Hermann, Karl Moritz and Blunsom, Phil},
  title     = {{The Role of Syntax in Vector Space Models of Compositional Semantics}},
  booktitle = {Proceedings of ACL},
  year      = {2013},
  month     = aug,
  url       = {http://www.karlmoritz.com/_media/hermannblunsom_acl2013.pdf}
}
```

Notes
====

This is development code and may not be fully functional. That said, the
**doctrain** and **dbltrain** programmes should work as expected and you should
be able to reproduce all the results from the papers mentioned above.

There are a number of auxiliary programmes to extract vectors and for file
conversion, as well as a number of programmes that use the underlying model for
different tasks such as question answering.

In particular the QA related code is still under development and included here
primarily in order to demonstrate how the model can easily be extended with
different training algorithms and modalities.


To Do
====

Dictionary/Model initialisation could be cleaned up!

There is some technical debt from the memory optimisation on the corpus side.
Principally this concerns the push_back overload for the corpus function - this
needs to be completed given more complex models.

Most models from the 2013 ACL paper have not yet been ported to the new
architecture (CCAE-A..D etc.). This should be done at some point. For now,
please refer to the ["oxcsvm" repository](https://github.com/karlmoritz/oxcvsm),
which features some of these models.
