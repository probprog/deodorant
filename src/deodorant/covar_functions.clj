(ns deodorant.covar-functions
  "Covariance functions for Deodorant."
  (:require [clojure.core.matrix :refer [matrix shape transpose mul add sub div join broadcast slice-views sqrt exp pow]]
            [deodorant.broadcast-functions :as bf]))

(defn- calc-d
  "Calculates scaled distance
   between points given log-rho and x-diff-squared

   Accepts: log-rho        - a vector
            x-diff-squared - a NxDxM matrix of squared distances or
                             NxD matrix of squared distances of old
                             points and new point

   Returns: d              - scaled distance between points
            d-squared      - squared scaled distance between points
            sep-squared    - squared seperations prior to summing
                             over a dimension"
  ; MEMOIZE ME
  [log-rho x-diff-squared]
  (let [rho-sq      (matrix (mapv (fn [x] (exp (* 2 x)))
                              log-rho))
        sep-squared (if (> (count (shape x-diff-squared)) 2)
                      (bf/scale-square-diff x-diff-squared (matrix rho-sq))
                      (div x-diff-squared rho-sq))
        d-squared   (bf/safe-sum sep-squared 1)]
    [(sqrt d-squared) d-squared sep-squared]))

(defn- exp-minus-sqrt3-d
  "Calculates  sqrt(3)d and exp(-sqrt(3)d) allowing memoization"
  ; MEMOIZE ME
  [d]
  (let [sqrt-3-d (mul (sqrt 3) d)]
    [sqrt-3-d (exp (sub 0 sqrt-3-d))]))

(defn- exp-minus-sqrt5-d
  "Calculates  sqrt(5)d and exp(-sqrt(5)d) allowing memoization"
  ; MEMOIZE ME
  [d]
  (let [sqrt-5-d (mul (sqrt 5) d)]
    [sqrt-5-d (exp (sub 0 sqrt-5-d))]))

(defn matern32-K
  "Covariance function for matern-32.

   Accepts: x-diff-squared - a NxDxN matrix of squared distances
                             or NxD matrix of squared distances of old points
                             and new point
            log-sig-f      - a scalar
            log-rho        - a vector

  Returns: A matrix K"
  ; MEMOIZE ME
  [x-diff-squared log-sig-f log-rho]
  (let [sig-f-sq                 (exp (* 2 log-sig-f))
        [d _ _]                  (calc-d log-rho x-diff-squared)
        [sqrt-3-d exp-m-sqrt3-d] (exp-minus-sqrt3-d d)]
    (mul sig-f-sq (mul (add 1 sqrt-3-d)
                       exp-m-sqrt3-d))))

(defn matern32-xs-z
  "Side covariance matrix for matern-32, i.e. vector k where
  k_i = kernel(x_i, z).

  Accepts:
  xs         - a NxD vector of vectors of xs
  z          - [Dx1] vector of new data point
  log-sig-f  - a scalar
  log-rho    - a vector

  Returns: A vector k sized N."
  [xs z log-sig-f log-rho]
  (let [sig-f-sq (exp (* 2 log-sig-f))
        xs-z-diff-squared (pow (sub (matrix xs) (matrix z)) 2)
        [d _ _] (calc-d log-rho xs-z-diff-squared)
        [sqrt-3-d exp-m-sqrt3-d] (exp-minus-sqrt3-d d)]
    (mul sig-f-sq (mul (add 1 sqrt-3-d)
                       exp-m-sqrt3-d))))

(defn matern52-K
  "Covariance function for matern-52.

   Accepts: x-diff-squared - a NxDxN matrix of squared distances
                             or NxD matrix of squared distances of old points
                             and new point
            log-sig-f      - a scalar
            log-rho        - a vector

  Returns: A matrix K"
  ; MEMOIZE ME
  [x-diff-squared log-sig-f log-rho]
  (let [sig-f-sq                 (exp (* 2 log-sig-f))
        [d d-sq _]               (calc-d log-rho x-diff-squared)
        [sqrt-5-d exp-m-sqrt5-d] (exp-minus-sqrt5-d d)]
    (mul sig-f-sq
         (add 1 sqrt-5-d (mul d-sq (/ 5 3)))
         exp-m-sqrt5-d)))

(defn matern52-xs-z
  "Side covariance matrix for matern-52, i.e. vector k where
  k_i = kernel(x_i, z).

  Accepts:
  xs         - a NxD vector of vectors of xs
  z          - [Dx1] vector of new data point
  log-sig-f  - a scalar
  log-rho    - a vector

  Returns: A vector k sized N."
  [xs z log-sig-f log-rho]
  (let [sig-f-sq (exp (* 2 log-sig-f))
        xs-z-diff-squared (pow (sub (matrix xs) (matrix z)) 2)
        [d d-sq _] (calc-d log-rho xs-z-diff-squared)
        [sqrt-5-d exp-m-sqrt5-d] (exp-minus-sqrt5-d d)]
    (mul sig-f-sq (mul (add (add 1 sqrt-5-d) (mul d-sq (/ 5 3)))
                       exp-m-sqrt5-d))))

(defn matern32-plus-matern52-K
  "Compound covariance function for matern-32 and matern-52.

   Accepts: x-diff-squared   - a NxDxN matrix of squared distances
            log-sig-f-32     - a scalar
            log-rho-32       - a vector
            log-sig-f-52     - a scalar
            log-rho-52       - a vector

   Returns: A matrix K"
   [x-diff-squared log-sig-f-32 log-rho-32 log-sig-f-52 log-rho-52]
   (let [K-32 (matern32-K x-diff-squared log-sig-f-32 log-rho-32)
         K-52 (matern52-K x-diff-squared log-sig-f-52 log-rho-52)]
     (add K-32 K-52)))

(defn- broadcast-as-required-in-grads
  "Calls broadcast-function-NxDxN-NxN when provided first matrix is NxDxN and
   just uses built in broadcast operations when NxD"
  [op M1 M2 & args]
  (let [n-dims (count (shape M1))]
    (if (> n-dims 2)
      (bf/broadcast-function-NxDxN-NxN op M1 M2 args)
      (transpose (op (transpose M1) M2)))))


(defn matern32-grad-K
  "Gradient for matern32.  Syntax as per matern32
   except returns a DxNxN array giving derivatives
   in the different directions.  The first entry
   of the first dimension corresponds to the derivative
   with respect to log-sig-f, with the others wrt
   log-rho"
  [x-diff-squared log-sig-f log-rho]
  (let [sig-f-sq                 (exp (* 2.0 log-sig-f))
        [d _ seq-squared]        (calc-d log-rho x-diff-squared)
        [sqrt-3-d exp-m-sqrt3-d] (exp-minus-sqrt3-d d)
        grad-K32-sig-f           (mul 2.0 (matern32-K x-diff-squared log-sig-f log-rho))
        [N _]                    (shape grad-K32-sig-f)
        grad-K32-rho             (-> (broadcast-as-required-in-grads mul seq-squared exp-m-sqrt3-d)
                                     (mul (* 3.0 sig-f-sq)))
        size-broad               (vec (concat [1] (shape grad-K32-sig-f)))]
    (join (broadcast grad-K32-sig-f size-broad) (slice-views grad-K32-rho 1))))

(defn matern52-grad-K
  "Gradient for matern52.  Syntax as per matern52
   except returns a DxNxN array giving derivatives
   in the different directions.  The first entry
   of the first dimension corresponds to the derivative
   with respect to log-sig-f, with the others wrt
   log-rho"
  [x-diff-squared log-sig-f log-rho]
  (let [sig-f-sq                 (exp (* 2.0 log-sig-f))
        [d _ seq-squared]        (calc-d log-rho x-diff-squared)
        [sqrt-5-d exp-m-sqrt5-d] (exp-minus-sqrt5-d d)
        grad-K52-sig-f           (mul 2.0 (matern52-K x-diff-squared log-sig-f log-rho))
        [N _]                    (shape grad-K52-sig-f)
        grad-K52-rho             (->> (add 1 sqrt-5-d)
                                      (mul exp-m-sqrt5-d)
                                      (broadcast-as-required-in-grads mul seq-squared)
                                      (mul (* (/ 5.0 3.0) sig-f-sq)))
        size-broad               (vec (concat [1] (shape grad-K52-sig-f)))]
    (join (broadcast grad-K52-sig-f size-broad) (slice-views grad-K52-rho 1))))

(defn matern32-plus-matern52-grad-K
  "Gradient of compound covariance function for matern-32 and matern-52.

   Accepts: x-diff-squared   - a NxDxN matrix of squared distances
            log-sig-f-32     - a scalar
            log-rho-32       - a vector
            log-sig-f-52     - a scalar
            log-rho-52       - a vector

   Returns: An DxNxN array grad-K giving derivatives
            in the different directions.  The first entry
            of the first dimension corresponds to the derivative
            with respect to log-sig-f, with the others wrt
            log-rho"
   [x-diff-squared log-sig-f-32 log-rho-32 log-sig-f-52 log-rho-52]
   (let [grad-K-32 (matern32-grad-K x-diff-squared log-sig-f-32 log-rho-32)
         grad-K-52 (matern52-grad-K x-diff-squared log-sig-f-52 log-rho-52)]
     (join grad-K-32 grad-K-52)))

(defn matern-for-vector-input
  "Converts a covariance function that takes pairs of log-sig-f
   and log-rho as inputs and converts them to one that accepts
   a vector with correctly ordered hyperparameters.

  Accepts: dim           - Dimension of data
           K             - Relevant kernel function

  Return: K-vec - the kernel function that now accepts x-diff-squared
                  followed by a vector"
  [dim K]
  (let [arrange-inputs   (fn [x]
                           (loop [in nil
                                  xl x]
                             (if (= xl [])
                               in
                               (recur (concat in [(first xl) (subvec xl 1 (inc dim))])
                                      (subvec xl (inc dim))))))]
    (fn [x-diff-squared hyper] (apply K x-diff-squared (arrange-inputs hyper)))))

(defn matern32-grad-z
  "Jacobian of side kernel matrix w.r.t. new data point z for Matern 32.
  If using a gradient based solver for the acquisition funciton, then
  needed for calculating derivative of Expected Improvement, EI(z), as outlined
  on page 3 of
  http://homepages.mcs.vuw.ac.nz/~marcus/manuscripts/FreanBoyle-GPO-2008.pdf.

  Accepts:
  xs-z-diff   - NxD matrix whose (i, j)th entry is x_ij - z_j
  log-sig-f   - scalar; parameter of kernel function
  log-rho     - D-dimensional parameter of kernel function

  Returns:
  [NxD] Jacobian of side kernel matrix w.r.t. new data point where
  (i, j)th entry is d(kernel(x_i, z)) / d(z_j)."
  [xs-z-diff log-sig-f log-rho]
  (let [sig-f-sq (exp (* 2 log-sig-f))
        rho-sq (exp (mul 2 log-rho))
        xs-z-diff-squared (pow xs-z-diff 2)
        [d _ _] (calc-d log-rho xs-z-diff-squared) ;; N-dimensional vector
        [_ ex3] (exp-minus-sqrt3-d d)]
    (-> (sub 0 xs-z-diff) ;; matrix of (z_j - x_ij) where i = 1:N, j = 1:D
        (div rho-sq) ;; matrix (NxD) of (z_j - x_ij) / rho_j
        transpose
        (mul ex3)
        transpose ;; matrix of exp(-\sqrt 3d_i) (z_j - x_ij) / rho_j
        (mul (* -3 sig-f-sq))))) ;; matrix of -3\sigma^2 exp(-\sqrt 3d_i) (z_j - x_ij) / rho_j

(defn matern52-grad-z
  "Jacobian of side kernel matrix w.r.t. new data point z for Matern 52.
  If using a gradient based solver for the acquisition funciton, then
  needed for calculating derivative of Expected Improvement, EI(z), as outlined
  on page 3 of
  http://homepages.mcs.vuw.ac.nz/~marcus/manuscripts/FreanBoyle-GPO-2008.pdf.

  Accepts:
  xs-z-diff   - NxD matrix whose (i, j)th entry is x_ij - z_j
  log-sig-f   - scalar; parameter of kernel function
  log-rho     - D-dimensional parameter of kernel function

  Returns:
  [NxD] Jacobian of side kernel matrix w.r.t. new data point where
  (i, j)th entry is d(kernel(x_i, z)) / d(z_j)."
  [xs-z-diff log-sig-f log-rho]
  (let [sig-f-sq (exp (* 2 log-sig-f))
        rho-sq (exp (mul 2 log-rho))
        xs-z-diff-squared (pow xs-z-diff 2)
        [d _ _] (calc-d log-rho xs-z-diff-squared)
        [sq5d ex5] (exp-minus-sqrt5-d d)]
    (-> (sub 0 xs-z-diff) ;; matrix of (z_j - x_ij) where i = 1:N, j = 1:D
        (div rho-sq) ;; matrix of (z_j - x_ij) / rho_j
        transpose
        (mul (add 1 sq5d))
        (mul ex5)
        transpose ;; matrix of exp(-\sqrt5d_i) * (1 + \sqrt5d_i) * (z_j - x_ij) / rho_j
        (mul (* (- (/ 5 3)) sig-f-sq)))))

(defn matern32-plus-matern52-grad-z
  "Jacobian of side kernel matrix w.r.t. new data point z for Matern 32 + Matern 52.
  If using a gradient based solver for the acquisition funciton, then
  needed for calculating derivative of Expected Improvement, EI(z), as outlined
  on page 3 of
  http://homepages.mcs.vuw.ac.nz/~marcus/manuscripts/FreanBoyle-GPO-2008.pdf.

  Accepts:
  xs-z-diff   - NxD matrix whose (i, j)th entry is x_ij - z_j
  log-sig-f   - scalar; parameter of kernel function
  log-rho     - D-dimensional parameter of kernel function

  Returns:
  [NxD] Jacobian of side kernel matrix w.r.t. new data point where
  (i, j)th entry is d(kernel(x_i, z)) / d(z_j)."
  [x-z-diff log-sig-f-32 log-rho-32 log-sig-f-52 log-rho-52]
  (add (matern32-grad-z x-z-diff log-sig-f-32 log-rho-32)
       (matern52-grad-z x-z-diff log-sig-f-52 log-rho-52)))
