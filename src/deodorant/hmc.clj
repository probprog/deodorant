(ns deodorant.hmc
  "Basic HMC sampler implementation."
  (:require [clojure.core.matrix :as mat
             :refer [matrix mul add sub]]
            [deodorant.helper-functions :refer [sample* normal]]))

(defn- sum
  [x & [dim & _]]
  (if (and dim (> dim 0))
    (reduce mat/add (mat/slice-views x dim))
    (reduce mat/add x)))

(defn- mean
  [x & [dim & _]]
  (let [dim (or dim 0)]
    (mat/div (sum x dim) (get (mat/shape x) dim))))

(defn- sq
  [x]
  (mat/mul x x))

(defn scale-vector
  "Scale a vector by a scalar"
  [v factor]
  (mapv (partial * factor) v))

(defn hmc-integrate
  "Preforms leap-frog integration of trajectory."
  [grad-u eps num-steps q p]
  (loop [q q
         p (mat/sub p (mat/mul 0.5 eps (grad-u q)))
         n 1]
    (if (< n num-steps)
      (let [q-new (mat/add q (mat/mul eps p))
            p-new (mat/sub p (mat/mul eps (grad-u q-new)))]
        (recur q-new p-new (inc n)))
      [q (mat/sub p (mat/mul 0.5 eps (grad-u q)))])))

(defn hmc-transition
  "Performs one Hamiltonian Monte Carlo transition update.

  Accepts functions u and grad-u with arguments [q], a parameter eps
  that specifies the integration step size, and a parameter num-steps
  that specifies the number of integration steps.

  Returns a new sample q."
  [u grad-u eps num-steps q-start]
  (let [[accept-prob
         q-end]      (try
                       (let [p-start (mat/matrix
                                      (map sample*
                                           (repeat (count q-start)
                                                   (normal 0 1))))
                             [q-end p-end] (hmc-integrate grad-u eps num-steps
                                                          q-start p-start)
                             k-start (* 0.5 (sum (sq p-start)))
                             k-end (* 0.5 (sum (sq p-end)))]
                         [(Math/exp (+ (- (u q-start) (u q-end))
                                       (- k-start k-end)))
                          q-end])
                       (catch Exception e
                         [0.0 nil]))
        ]
    (if (> accept-prob (rand))
      q-end
      q-start)))

(defn hmc-chain
  "Performs Hamiltonian Monte Carlo to construct a Markov Chain

  Accepts functions u and grad-u with arguments [q], a parameter eps
  that specifies the integration step size, and a parameter num-steps
  that specifies the number of integration steps.

  Returns a lazy sequence of samples q."
  [u grad-u eps num-steps q-start]
  (let [q-next (hmc-transition u grad-u eps num-steps q-start)]
    (lazy-seq
     (cons q-next (hmc-chain u grad-u eps num-steps q-next)))))

(defn burn-in-and-thin
  "Takes the output of a markov chain, removes a number of burn-in samples
  and thins

  Accpets: burn-in-proportion
  thin-rate
  samples

  Retruns: samples(n-start:thin-rate:end)
  where n-start = (int (* (count samples) burn-in-proportion))"
  [burn-in-proportion thin-rate samples]
  (let [n-burn-in (int (Math/ceil (* (count samples) burn-in-proportion)))
        samples (take-nth thin-rate (vec (drop n-burn-in samples)))]
    samples))

(defn collapse-identical-samples
  "Takes an unweighted collection of samples and returns the unique values
  allong with a vector of the number of times they occured.  Ordering
  correspondings to the times of first apperance"
  [samples]
  (let [freq (frequencies samples)
        weights (mapv second freq)
        weights (scale-vector weights (/ 1 (reduce + weights)))
        unique-samples (if (vector? samples)
                         (mapv first freq)
                         (map first freq))
        ]
    [unique-samples weights]))
