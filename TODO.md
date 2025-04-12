TODO
====

This document is a primarily a wishlist and ideas list, and only secondarily
(if at all) a development roadmap.


### New Methods, Algorithms, and exploratory directions:

+ Distributed GPU enabled batch whitening.
+ Distributed GPU enabled FastICA.
+ ICA by time lagged covariance matrices.
+ ICA using incremental gradient algorithms (minibatch learning).

More vaguely (algorithmic directions to explore implementing):

+ Gaussian Processes (esp. gpytorch - measure performance)
+ Conditional Neural Processes (CNPs)
+ Neural Network Gaussian Processes (NNGPs)
+ LiNGAM and its many variants


### Better code, in-code docs, tests, etc.:

+ Provide example pdfs or ipy notebooks illustrating our plotting functions.
+ Clean up (merge and purge) our stunning array of different methods to
  calculate out-of-core descriptive statistics (i.e., harvest stats over
  datasets too large for memory: currently there are versions backed by numpy,
  pandas, and pytorch. If a unified implementation isn't possible, at least a
  unified API or set of naming conventions may be).
+ Ensure truthfulness and utility of code comments and docstrings.
+ Test and visualize curve construction functions.


### Better documents:

+ Produce a set of tutorials introducing prospective students to the basic
  processes and ideas of the lab.
  - intro to OMOP data selection and cleaning
  - intro to longitudinal curve functions and transformations
  - intro to data whitening and PCA generally
  - intro to ICA
  - summary of downstream usages in training classifiers, predictors, etc
+ Provide an abbreviated annotated bibliography for the most important works we
  rely on - the ICA book and related papers, etc.
+ Provide an annotated bibliography of papers written by lab members or
  associates using our methodologies or products.
+ Provide a list of links for other resources (OHDSI website, etc).
+ Provide a tutorial (or set of tutorials) outlining general good practice for
  performance-aware software development. There are many possible topics here:
  machine topology and environment, multiprocessing, basic data structures and
  how to use them, how to store data in a format that balances ease of use with
  performance, etc.


### Misc.:

+ Matricize or vectorize `fast_intensity` and include here (if we're going to
  continue using this algorithm: also, need to write a proper paper outlining
  the method).
+ Port the calibration tools here; take the opportunity to write proper tests,
  example visualizations, and review the code for performance and correctness.
