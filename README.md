# Deodorant: Solving the problems of BO

Deodorant is a Bayesian optimization package with three core features:

1. Domain scaling to exploit problem independent GP hyperpriors

2. A non-stationary mean function to allow unbounded optimization

3. External provision of the acquisition function optimizer so that this can incorporate the constraints of the problem (inc equality constraints) and ensure that no invalid points are evaluated.
\end{enumerate}
  
The main intended use of the package at present is as the BO component for [BOPP](https://github.com/twgr/bopp) (Bayesian Optimiation for Probabilistic Programs. Rainforth T, Le TA,
  van de Meent J-W, Osborne MA, Wood F. In NIPS 2016) which provides all the
  required inputs automatically given a program.  Even when the intention is
  simply optimization, using BOPP rather than Deodorant directly is currently
  recommended.  The rational of providing Deodorant as its own independent
  package is to seperate out the parts of BOPP that are Anglican dependent and
  those that are not.  As such, one may wish to intergrate Deodorant into
  another similar package that provides all the required inputs.
  
  For details on the working of Deodorant, the previously referenced paper and
  its supplementary material should be consulted.
  
## Installation ##
  
Though Deodorant has no direct dependency on Anglican, it has the same requirements in terms
of java, Leiningen etc and so we refer the reader to http://www.robots.ox.ac.uk/~fwood/anglican/usage/index.html
and recommend that users follow section 2 in the user start up guide.  The above link is also a good starting
point for further links on Clojure etc.

## License ##

Copyright © Tom Rainforth, Tuan Anh Le, Jan-Willem van de Meent

Deodorant is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Deodorant is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the [GNU General Public License](gpl-3.0.txt) along with Deodorant.  If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).
