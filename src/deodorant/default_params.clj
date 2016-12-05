(ns deodorant.default-params
  "Helper functions for Deodorant."
  (:require [clojure.core.matrix :as mat]
            [deodorant.hyper-priors :as hyper]
            [deodorant.broadcast-functions :as bf]))

(defn default-double-matern-hyperprior
  "Sets up a default hyperprior based on the composition
   of a matern-32 and a matern-52 kernel.  Accepts the
   dimensionality of the input space dim and returns a hash
   map with fields :sampler, :log-p and :grad-log-p.  Each
   of these operate on
   [log-sig-n [log-sig-f-32 log-rho-32-dim1 log-rho-32-dim2 ....]
              [log-sig-f-52 log-rho-52-dim1 log-rho-52-dim2 ....]].
   :sampler returns a set of samples of this form.
   :log-p returns a scalar given a set of parameters
   :grad-log-p returns a nested vector of the same size as sampler
               does corresponding to the gradient of that hyperparameter"
  [dim]
  (let [log-sig-noise-mean       -5
        log-sig-noise-std-dev    2
        log-sig-f-mean-32        -7
        log-sig-f-std-32         0.5
        log-rho-mean-32          -1.5
        log-rho-std-dev-32       0.5
        log-sig-f-mean-52        -0.5
        log-sig-f-std-52         0.15
        log-rho-mean-52          -1
        log-rho-std-dev-52       0.5
        dist-h-32   (hyper/constant-length-distance-hyperprior
                      dim log-sig-f-mean-32 log-sig-f-std-32 log-rho-mean-32 log-rho-std-dev-32)
        dist-h-52   (hyper/constant-length-distance-hyperprior
                      dim log-sig-f-mean-52 log-sig-f-std-52 log-rho-mean-52 log-rho-std-dev-52)]
    (hyper/compose-hyperpriors
         dim log-sig-noise-mean log-sig-noise-std-dev dist-h-32 dist-h-52)))

(defn- bump-function-1d
  "A bump function used in the gp mean.  Takes a radius from
   an original point, a ridge-value, an inf-value and exponent
   and returns a function evaluation.  Note r should be positive.
   When r<ridge-value then 0 is returned.  When r>inf-value then
   -inf is returned.  Otherwise exponent*log(inf-value-r)+a*r+c is returned
   where a and c are set to give the required ridge-value.

  Accepts [ridge-value inf-value exponent r]
  Returns scalar"
  [ridge-value inf-value exponent r]
  (let [a (- (/ (* exponent (Math/pow (- ridge-value inf-value) (dec exponent)))
                (Math/pow (- ridge-value inf-value) exponent)))
        c (- (+ (* exponent (Math/log (- inf-value ridge-value)))
                (* a ridge-value)))]
    (if (or (< r 0) (> r inf-value))
      (- (/ 1.0 0.0))
      (if (< r ridge-value)
        0
        (+ (* exponent (Math/log (- inf-value r)))
           (* a r)
           c)))))

(defn- bump-function
  "Calculates the radius of a point and uses it as the input
   to bump-function-1d

  Accepts [ridge-value inf-value exponent x]
  Returns scalar"
  [ridge-value inf-value exponent x]
  (let [r (mat/sqrt (bf/safe-sum (mat/pow x 2) 1))]
    (mapv #(bump-function-1d ridge-value inf-value exponent %) r)))

(defn default-mean-fn-form [dim]
  (partial bump-function 1 (* 1.5 (Math/sqrt dim)) 1))

(defn default-hmc-initializer [n-chains hyperprior]
  ((:sampler hyperprior) n-chains))
