(ns deodorant.acq-functions
  "Acquisition functions for Deodorant."
  (:require [clojure.core.matrix :as mat]
            [deodorant.broadcast-functions :refer [safe-sum]]
            [deodorant.helper-functions :refer [erf observe* normal]]))

(defn- normal-cdf
  "Normal cumulative density function."
  [mean stdd x]
  (let [xp (/ (- x mean) (* stdd (Math/sqrt 2)))]
    (* 1/2 (+ 1 (if (> xp 0)
                  (erf xp)
                  (- (erf (- xp))))))))

(defn expected-improvement
  "Expected improvement acquisition function with some trickery
   to overcome underflow when very low.  Specifically, we
   revert to using log(UCB) with v=2 as the acquisition function
   if log(EI/exp(UCB))<-8.

  Accepts:
    xi - option for expected improvement (scalar)
    mu-best - best posterior mean evaluated at old data points (scalar)
    gp-predicter - function that returns [mu var]
                   from the weighted gp ensemble posterior given
                   a single point to evaluate x.  See
                   create-weighted-gp-predict-with-derivatives-function*
    x* - new point to evaluate (D length vector)

  Returns:
    EI - log value of expected improvement at x*"
  [xi mu-best gp-predicter x*]
  (let [[mu sig] (gp-predicter x*)
        u (mat/div (mat/sub mu (+ mu-best xi)) sig)
        phi (mapv #(mat/exp (observe* (normal 0 1) %)) u)
        Phi (mapv #(normal-cdf 0 1 %) u)
        EI (mat/mul sig (mat/add (mat/mul u Phi) phi))
        EI (mat/emap (fn [x] (if (Double/isNaN x)  0 (max 0 x))) EI)
        UCB (mat/add mu (mat/scale sig 2))
        ratio (mat/div EI (mat/exp UCB))
        ratio (mat/emap #(if (< % 1e-8) 1e-8 %)
                    ratio)
        ;; The UCB component will cancel out below unless
        ;; the cap on the ratio is in force, when it becomes
        ;; (+ UCB (log 1e-8))
        logEI (mat/add UCB (mat/log ratio))
        EI (mat/exp logEI)]
    EI))

(defn integrated-aq-func
  "Calculates the integrated acquisition function from a base
   acquisition function and a weighted sum of GP predictors.
   This is as per Snoek et al, NIPS 2012.

  Accepts:
    base-aq-func - Function which takes as inputs a gp-predictor
                   and a point to evaluate, returning a utility of
                   evaluating that point.
    gp-predicters - A collection of gp-prediction functions.  Each
                    takes a point to evaluate an returns [mu sig]
    gp-weights - Vector of weights for each gp-predictor.  The
                 acquisition function values will be weighted by
                 these weights
    x* - The point to evaluate (D length vector)
  Returns:
    The integrated acquistion function at x*"
  [base-aq-func mu-bests gp-predicters gp-weights x*]
  (let [acq-vals (mapv (fn [w m p] (mat/mul w
                                          (base-aq-func m p x*)))
                    gp-weights
                    mu-bests
                    gp-predicters)]
  (safe-sum acq-vals 0)))
