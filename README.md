# Deodorant: Solving the problems of BO

Deodorant is a Bayesian optimization package with three core features:

1. Domain scaling to exploit problem independent GP hyperpriors
2. A non-stationary mean function to allow unbounded optimization
3. External provision of the acquisition function optimizer so that this can incorporate the constraints of the problem (inc equality constraints) and ensure that no invalid points are evaluated.

The main intended use of the package at present is as the BO component for [BOPP](https://github.com/probprog/bopp):

Rainforth, T., Le, T. A., van de Meent, J.-W., Osborne, M. A., & Wood, F. (2016). Bayesian Optimization for Probabilistic Programs. In Advances in Neural Information Processing Systems.

```
@incollection{rainforth2016bayesian,
    title = {Bayesian Optimization for Probabilistic Programs},
    author = {Rainforth, Tom and Le, Tuan Anh and van de Meent, Jan-Willem and Osborne, Michael A and Wood, Frank},
    booktitle = {Advances in Neural Information Processing Systems 29},
    pages = {280--288},
    year = {2016},
    url = {http://papers.nips.cc/paper/6421-bayesian-optimization-for-probabilistic-programs.pdf}
}
```

which provides all the required inputs automatically given a program.  Even when the intention is simply optimization, using BOPP rather than Deodorant directly is currently recommended.  The rational of providing Deodorant as its own independent package is to separate out the parts of BOPP that are Anglican dependent and those that are not.  As such, one may wish to integrate Deodorant into another similar package that provides all the required inputs.

For details on the working of Deodorant, the previously referenced paper and its supplementary material should be consulted.

## Installation ##

To use Deodorant in your own [Leiningen](http://leiningen.org/) projects, just include the dependency in your `project.clj`:
```
(defproject foo
  ...
  :dependencies [...
                 [deodorant "0.1.0"]
                 ...])
```

In your Clojure files, remember to require functions from `core.clj`, e.g.:
```
(ns bar
  (require [deodorant.core :refer :all]))
```
The full documentation can be found [here](https://probprog.github.io/deodorant/). Checkout [core/deodorant](https://probprog.github.io/deodorant/deodorant.core.html#var-deodorant) in particular.

Though Deodorant has no direct dependency on Anglican, it has the same requirements in terms
of java, Leiningen etc and so we refer the reader to http://www.robots.ox.ac.uk/~fwood/anglican/usage/index.html
and recommend that users follow section 2 in the user start up guide.  The above link is also a good starting
point for further links on Clojure etc.

## License ##

Copyright © Tom Rainforth, Tuan Anh Le, Jan-Willem van de Meent, Michael Osborne and Frank Wood

Deodorant is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Deodorant is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the [GNU General Public License](gpl-3.0.txt) along with Deodorant.  If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).
