# barber
The Barber cuts and sweeps galaxies into tomographic bins

## Motivation
The redshift dependence of cosmological probes can better constrain cosmological models and parameters.
Such a slicing of a galaxy sample on the basis of redshift would enable a tomographic analysis, where the parameters defining the slicing scheme are a choice by the researcher.
However, for a photometric survey like LSST, the redshift is an unobserved variable, precluding straightforward division of a galaxy sample using that parameter.
One can imagine many approaches to deriving subsamples of galaxies, and [tomo_challenge](https://github.com/LSSTDESC/tomo_challenge) aims to compare a variety of methods to identify which one(s) are sufficiently promising to be incorporated into the LSST-DESC analysis pipeline(s).
This repo is home to one or more proposed solutions to this problem.

## Scope
The simplest possible approach to dividing galaxies is to use deterministic cuts in the space of observable quantities.
The Barber optimizes the values of the parameters defining these cuts for a sample of galaxies based on a given metric.
This work will enable us to answer the following question: "How well can we constrain the cosmological parameters in a 3x2pt analysis with tomographic bins defined by nothing more sophisticated than simple cuts on observable quantities of the galaxy sample?"
Based on the requirements of a given survey, the answer to this question can determine whether it needs a more sophisticated analysis pipeline.
