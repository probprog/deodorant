(ns deodorant.gp-toolbox
  "GP functions for training, testing, calculating the marginal likelihood
  and its derivatives."
  (:require [clojure.core.matrix :refer [matrix identity-matrix dot shape
                                         inverse transpose trace outer-product
                                         mul mmul add sub div join broadcast
                                         slice-views exp log pow sqrt]]
            [clatrix.core :as clx]
            [deodorant.helper-functions :refer [sample* mvn]]
            [deodorant.broadcast-functions :as bf]))

(defn subtract-mean-fn
  "Subtracts the GP mean. Accepts a mean-fn from arguments [x] to a
  scalar y, along with a collection of points [x y]. Returns a vector
  of values (- y (mean-fn x))."
  [mean-fn x y]
    (sub y (mean-fn x)))

(defn gp-train
  "Trains a gp, returns L and psi - the lower triangular matrix
  and vector of differences required for prediction.

  Accepts: cov-fn - a function taking inputs of x-diff-sq and
                    a vector of hyperparameters (not including noise
                    parameter and therefore corresponding to (rest alpha))
           x-diff-sq - a NxDxN matrix of squared distances of points
           y-bar  - observations minus the value of the prior mean function
                    at those points
           alpha  - a vector of hyperparameters ordered as
                   [log-sig-n log-sig-f-k1 log-rho-k1-dim-1 log-rho-k1-dim-2 ... log-sig-f-k2 ...]

  Returns:  L     - lower triangular matrix used for gp prediction
            psi   - inv-K times y-bar"
  ; MEMOIZE ME
  [cov-fn x-diff-sq y-bar alpha]
  (let [_ (if (not (every? (fn [x] (< x 10)) (rest alpha)))
            (let [;_ (print :alphafail alpha)
                  ] (throw (Exception. "Alphas are at silly values.  Throwing exception before clatrix kills java.")))
            nil)
        K-no-noise (cov-fn x-diff-sq (vec (rest alpha)))
        [N D _]    (shape x-diff-sq)
        sig-n-sq   (exp (* 2 (first alpha)))
        ;; A little naughty but for numerical stability add a little onto sig-n-sq
        ;; note we don't bother checking sig-n at the start because of this
        sig-n-sq (+ sig-n-sq 0.00000001)
        K          (add K-no-noise (mul sig-n-sq (identity-matrix N)))
        L          (try
                      (transpose (matrix (clx/cholesky (clx/matrix K))))
                      (catch Exception e
                        (println :calc-L-failed-with-error e)
                        (println :alpha alpha)
                        (println :K K)
                        (flush)
                        (throw (Exception. "calculating L failed due to numerical instability"))))
        psi        (try
                     (clx/solve (clx/matrix (transpose L)) (clx/solve (clx/matrix L) (clx/vector y-bar)))
                     (catch Exception e
                       (println :calc-psi-failed-with-error e)
                       (println :alpha alpha)
                       (println :K K)
                       (println :L L)
                       (flush)
                       (throw (Exception. "calculating psi failed due to numerical instability"))))]
    [(matrix L) (matrix (vec psi))]))


(defn gp-log-likelihood
  "Calculates the gp-log-likelihood given L, psi and y-bar"
  ; MEMOIZE ME
  [L psi y-bar]
  (sub (mul -0.5 (mmul (transpose y-bar) psi))
       (trace (log L))
       (* 0.5 (count y-bar) (log (* 2 Math/PI)))))

(defn gp-grad-log-likelihood
  "Calculates the gradient of the gp-log-likelihood with
  respect to the hyperparameters.

  Accepts:  grad-cov-fn  - Function to return grad of covariance func wrt
                           the hyperparameters returned as DxNxN matrix
            x-diff-sq    - see gp-train
            L            - see gp-train
            psi          - see gp-train"
  ; MEMOIZE ME
  [grad-cov-fn x-diff-sq alpha L psi]
  (let [grad-k-no-noise   (grad-cov-fn x-diff-sq (vec (rest alpha)))
        [N _ _]           (shape x-diff-sq)
        grad-k            (join (broadcast (mul 2
                                                (exp (* 2 (first alpha)))
                                                (identity-matrix N))
                                           [1 N N])
                                grad-k-no-noise)
        inv-L             (inverse L)
        inv-K             (mmul (transpose inv-L) inv-L)
        psi-psi-T         (outer-product psi psi)
        psi-psi-T-minus-inv-K (sub psi-psi-T inv-K)]
    (mul 0.5 (mapv (fn [x]
                     (trace (mmul psi-psi-T-minus-inv-K x)))
                   grad-k))))

(defrecord trained-gp-obj
  [prior-mean-fn      ; Prior function for mean.  Calcs are done using a zero mean gp and this added back at the end
   x-obs              ; Observed x values
   L                  ; Lower triangular decomposition of K
   psi                ; inv-K times y-bar
   inv-K              ; Inverse of covariance function for trained points
   inv-L              ; Inverse of cholesky decomposition of K
   sigma-n            ; Standard deviation of noise
   log-likelihood     ; log likelihood of the GP
   x*-diff-fn         ; Function that takes an array of new points and returns a NxDxM of seperations to the observed points
   k*-fn              ; Function to calculate covariance of (M) new points to (N) old points, returns NxM matrix
                      ;  overloaded to work on both a single point (e.g. [1 0.5]) and array of points (e.g. [[1 0.5]] or [[1 0.5] [-0.2 0.3]])
   k-new-fn           ; Function to calculate covariance between (M) new points, returns MxM matrix
                      ;  overloaded to work on both a single point (e.g. [1 0.5]) and array of points (e.g. [[1 0.5]] or [[1 0.5] [-0.2 0.3]])
   marginal-prior-var ; Result of calling k-new-fn on a single point (i.e. variance with itself under the prior)
   ;; Optional fields that are not currently used in BOPP.
   grad-prior-mean-fn-x ; Grad of the prior-mean-fn w.r.t x, returns different differentials as first dimension.
   grad-k*-fn         ; Function giving the gradient of k* with respect to the different dimensions of the proposed point
                      ;  overloaded to work on both a single point (e.g. [1 0.5]) and array of points (e.g. [[1 0.5]] or [[1 0.5] [-0.2 0.3]])
   ])

(defn create-trained-gp-obj
  "Created a trained-gp-obj record that is used for efficient
  prediction of gp at future points.

  Accepts:     prior-mean-func
               cov-fn                  Same form as sent to gp train
               points                  Vector of pairs of [x y] observations
               alpha                   Vector of hyperparameters in same form as sent to gp train

  Optional Inputs:
               grad-prior-mean-fn-x    Gradient of prior mean function.  Needed for some derivative
                                       calculations but not for basic use.
               grad-prior-cov-fn-x     Gradient of prior covariance function.  Needed for some derivative
                                       calculations but not for basic use.

  Returns:     trained-gp-obj

  Note that current usage of BOPP does not set these optional inputs.  They would be needed for anything
  that requires taking gradients with respect to the GP inputs, for example solving the acquisition
  function using gradient methods and left in for potential future convenience / other toolbox use."
  [prior-mean-fn cov-fn x-obs y-obs alpha
   & [grad-prior-mean-fn-x grad-prior-cov-fn-x]]
  (let [;; For details on these terms see the comments in trained-gp-obj
        y-bar             (subtract-mean-fn prior-mean-fn x-obs y-obs)
        x-diff-sq         (matrix (bf/square-diff x-obs))
        sigma-n           (exp (first alpha))
        hypers            (vec (rest alpha))
        [L psi]           (gp-train cov-fn x-diff-sq y-bar alpha)
        log-likelihood    (gp-log-likelihood L psi y-bar)
        x*-diff-fn        (fn [x*]
                            (bf/safe-broadcast-op sub x-obs (matrix x*))) ; Returns a N-OBSxDxN* array
        k*-fn             (fn [x*-diff]
                            (cov-fn (pow x*-diff 2) hypers))
        k-new-fn          (fn [x*]
                            (cov-fn (pow (bf/safe-broadcast-op sub (matrix x*) (matrix x*)) 2); Returns a N*xDxN* array
                                    hypers))
        [N D]             (shape x-obs)
        marginal-prior-var (first (k-new-fn (vec (repeat D 1)))) ; Variance of a single point under the prior
        inv-L             (inverse L)
        inv-K             (mmul (transpose inv-L) inv-L)
        grad-k*-fn        (if (= nil grad-prior-cov-fn-x)
                            nil
                            (fn [x*-diff]
                              (grad-prior-cov-fn-x x*-diff hypers)))]
    (->trained-gp-obj
     prior-mean-fn x-obs L psi inv-K inv-L sigma-n log-likelihood
     x*-diff-fn k*-fn k-new-fn marginal-prior-var grad-prior-mean-fn-x grad-k*-fn)))

(defn gp-predict-mu-sig
  "Makes gp predictions for mu and marginal standard deviation
   for multiple points simultaneously.

   Accepts   gp        - of type trained-gp-obj
             x*        - new points to evaluate   (MxD matrix)

   Returns   mu        - predicted means (M length vector)
             sig       - marginal predicted standard deviations"
  [gp x*]
  (let [k*       ((:k*-fn gp) ((:x*-diff-fn gp) x*))
        psi      (:psi gp)
        mu       (add ((:prior-mean-fn gp) x*) (mmul (transpose k*) psi))
        v        (mmul (:inv-L gp) k*)
        sig      (sqrt (sub (:marginal-prior-var gp) (bf/safe-sum (pow v 2) 0)))]
    [mu sig]))

(defn gp-predict-mu-cov
  "Makes gp predictions for mu and
   covariance for multiple points simultaneously.

   Accepts   gp        - of type trained-gp-obj
             x*        - new points to evaluate   (MxD matrix)
             & args    - if (first args) is true then the full covariance matrix
                         is returned instead of just the marginal variance

   Returns   mu        - predicted means (M length vector)
             cov       - (MxM matrix) corresponding to the covariance between the prediction points"
  [gp x* ]
  (let [k*       ((:k*-fn gp) ((:x*-diff-fn gp) x*))
        psi      (:psi gp)
        mu       (add ((:prior-mean-fn gp) x*) (mmul (transpose k*) psi))
        v        (mmul (:inv-L gp) k*)
        cov      (sub ((:k-new-fn gp) x*) (mmul (transpose v) v))]
    [mu cov]))


(defn gp-mixture-mu-sig
  "Calculates the mean and standard deviation from a weighted
   sum of gps, i.e. a gp mixture model.  Note that the resulting
   distribution is not a Gaussian, (the marginals are mixtures of
   Gaussians) but the mean and covariance is still analytic.

   Accepts:
    gp-predictors - A collection of gp prediction functions
    gp-weights - The relative weights of the gps
    xs - Positions to calculate the estimates at

   Returns:
    mus - The means of the points
    sigs - The standard deviations of the points"
  [gp-predictors gp-weights xs]
    (let [
        ;; Though the weighted sum of gps is not a GP itself, the means
        ;; still add is if they were
        mu-sigs (mapv #(% xs) gp-predictors)

        mu-samples (transpose (mapv first mu-sigs))
        sig-samples (transpose (mapv second mu-sigs))

        mus (bf/safe-sum (mul mu-samples gp-weights) 1)

        ; eq A.1.8 in mikes thesis
        sigs (sqrt (sub (bf/safe-sum (mul (add (pow sig-samples 2)
                                                       (pow mu-samples 2))
                                              gp-weights)
                                    1)
                                (pow mus 2)))]
      [mus sigs]))

;;;;;;;;;;;;;;  These are not actively used by BOPP but are useful GP functions ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


(defn gp-predict-with-derivatives
  "Makes gp predictions for mu, var, grad-mu and grad-var
  given a gp-object and a single query point

  Accepts    gp        - of type trained-gp-obj
             x*        - new points to evaluate   (D length vector)

  Returns    mu        - predicted means                        (vector of length 1)
             var       - marginal predicted vartion             (vector of length 1)
             grad-mu   - derivative of the mean with respect to the dimensions of predicted points. (D length vector)
             grad-var  - derivative of the variance with respect to the dimensions of predicted points. (D length vector)"
  [gp x*]
  (let [x*-diff  ((:x*-diff-fn gp) x*)
        k*       ((:k*-fn gp) x*-diff)
        psi      (:psi gp)
        mu       (add ((:prior-mean-fn gp) [x*]) (dot k* psi))
        v        (mmul (:inv-L gp) k*)
        var      (sub ((:k-new-fn gp) x*) (dot v v))
        grad-k*-t  ((:grad-k*-fn gp) x*-diff)
        grad-mu  (add ((:grad-prior-mean-fn gp) x*) (mmul (transpose grad-k*-t) psi))
        grad-var (sub (mmul 2 (:inv-K gp) k* grad-k*-t))]
    [[mu] [var] grad-mu grad-var]))

(defn gp-sample
  "Generates samples from a trained-gp-obj

   Accepts   gp          - of type trained-gp-obj
             x*          - points to evaluate   (MxD matrix)
             n-samples   - number of samples to generate

   Returns   f*          - sampled values for gp output (n-samples x M matrix).  Note that the y ~ N(f*,sigma-n^2)
             dist-f*     - mvn distribution object that allows for further efficient sampling if required"
  [gp x* n-samples]
  (let [[mu cov] (gp-predict-mu-cov gp x* true)
        dist-f*    (mvn mu cov)
        f*         (matrix (repeatedly n-samples #(sample* dist-f*)))]
    [f* dist-f*]))

(defn convert-output-to-std-dev
  "Takes the output of a gp prediction function and converts the variance
   terms to standard deviation terms for both original derivative"
  [mu var & args]
  (let [sig       (sqrt var)]
    (if (empty? args)
      [mu sig]
      (let [grad-mu   (first args)
            grad-sig (and (second args) (div (second args) (* 2 (first sig))))]
        [mu sig grad-mu grad-sig]))))
