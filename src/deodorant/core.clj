(ns deodorant.core
  "Core functions of Deodorant."
  (:refer-clojure :exclude [rand rand-nth rand-int])
  (:require [clojure.core.matrix :as mat]
            [clojure.core.matrix.operators :as mop]
            [taoensso.tufte :as tufte :refer (defnp profiled profile)]
            [deodorant.gp-toolbox :as gp]
            [deodorant.acq-functions :as acq]
            [deodorant.scaling-functions :as sf]
            [deodorant.hmc :as hmc]
            [deodorant.broadcast-functions :as bf]
            [deodorant.covar-functions :as cf]
            [deodorant.default-params :as dp]
            [deodorant.helper-functions :refer [indexed-max cartesian mean]]
            [clojure-csv.core :refer :all]))

(defn- hyper-prior-log-posterior
  "Gives the pdf for the posterior over gp hyperparameters

   Accepts: hyper-prior-log-p         log-pdf for the hyperprior
            cov-fn                    for of covariance for gp without hyperparameters set
            x-diff-sq
            y-bar
            alpha                     current value of the hyperparameters

   Returns: log-p for the posterior including terms from gp"
   [hyper-prior-log-p cov-fn x-diff-sq y-bar alpha]
  (let [[L psi] (gp/gp-train cov-fn x-diff-sq y-bar alpha)
        lik-gp  (gp/gp-log-likelihood L psi y-bar)]
    (+ lik-gp (hyper-prior-log-p alpha))))

(defn- grad-hyper-prior-log-posterior
  "Gives derivative of the pdf for the posterior over gp hyperparameters
   with respect to each of the hyperpriors.

   Accepts: hyper-prior-grad-log-p    gradient of log-pdf for the hyperprior
            cov-fn                    for of covariance for gp without hyperparameters set
            grad-cov-fn-hyper         gradient of covariance function wrt hyperparameters
            x-diff-sq
            y-bar
            alpha                     current value of the hyperparameters

   Returns: grad-log-p for the posterior including terms from gp"
   [hyper-prior-grad-log-p cov-fn grad-cov-fn-hyper x-diff-sq y-bar alpha]
  (let [[L psi]          (gp/gp-train
                           cov-fn x-diff-sq y-bar alpha)
        grad-log-lik-gp  (gp/gp-grad-log-likelihood
                           grad-cov-fn-hyper x-diff-sq alpha L psi)]
    (mat/add grad-log-lik-gp (hyper-prior-grad-log-p alpha))))

(defn infer-gp-hyper
  "Takes a mean-fn, a cov-fn (with unset hyperparameters), a
   series of points and hyper-prior and returns a weighted set
   of hyperparameter samples using a HMC sampler"
  [X Y &
   {:keys [mean-fn cov-fn grad-cov-fn-hyper gp-hyperprior hmc-step-size
           hmc-num-leapfrog-steps hmc-num-steps hmc-num-chains
           hmc-burn-in-proportion hmc-max-gps]
    :or {}}]
  (let [;; working with a zero mean GP and adding mean in later
        y-bars         (gp/subtract-mean-fn mean-fn X Y)

        ;; common term used everywhere
        x-diffs-sq     (bf/square-diff X)

        ;; function and its gradient for the log posterior of the gp hyperparameters
        f              (partial hyper-prior-log-posterior (:log-p gp-hyperprior) cov-fn x-diffs-sq y-bars)
        grad-f         (partial grad-hyper-prior-log-posterior (:grad-log-p gp-hyperprior) cov-fn grad-cov-fn-hyper x-diffs-sq y-bars)

        ;; hmc needs negative of this
        u              (fn [alpha] (- (f alpha)))            ; MEMOIZE ME
        grad-u         (fn [alpha] (mat/sub (grad-f alpha))) ; MEMOIZE ME

        ;; select the points at which to initialize the hmc chains
        alpha-starts   (map vec (dp/default-hmc-initializer hmc-num-chains gp-hyperprior))

        start-grads (mapv grad-u alpha-starts)
        abs-start-grads (mapv mat/abs start-grads)
        mean-abs-start-grads (mean abs-start-grads 0)
        eps-scaled (mop/min (mat/div hmc-step-size mean-abs-start-grads)
                            (vec (repeat (first (mat/shape mean-abs-start-grads))
                                         hmc-step-size)))

        call-hmc-sampler   (fn [q-start]
                             (take hmc-num-steps (hmc/hmc-chain u grad-u eps-scaled hmc-num-leapfrog-steps q-start)))
        thin-rate 1 ; Now using max-n-gps instead
        [alpha-samples weights] (->> (pmap (fn [x]
                                             (hmc/burn-in-and-thin
                                              hmc-burn-in-proportion
                                              thin-rate
                                              (call-hmc-sampler x)))
                                           alpha-starts)
                                     (reduce concat)
                                     vec
                                     hmc/collapse-identical-samples)
        i-keep (take hmc-max-gps (shuffle (range (count weights))))
        alpha-samples (mapv #(nth alpha-samples %) i-keep)
        weights (mapv #(nth weights %) i-keep)
        weights (hmc/scale-vector weights (/ 1 (reduce + weights)))
        alpha-particles (take hmc-max-gps
                              (shuffle (mapv (fn [a b] [a b])
                                             alpha-samples weights)))]
    alpha-particles))

(defn bo-acquire
  "Performs the acquisition step in Bayesian optimization. Accepts a
  sequence of points [x y] and returns the next point x-next to
  evaluate, the index of the point from those previously evaluated
  that is considered to be the best, and the predicted mean and std-dev
  of the evaluated points.  Note this std-dev is in the estimate of the
  'true' function value at this point and does not include the noise involved
  in evaluating this function.

  Accepts:
    - [[x1 y1 other] ...] vector of points where each point [xi yi other] consists of
      + xi - D-dimensional vector of points evaluated so far and
      + yi - scalar function evaluations of an expensive function at xi.
      + other - some other information for that particular point, e.g. predicts from prob prog.
    - acq-query - source code of acquisition query; used for BAMC-style optimisation.
    - options - optional options for the BO algorithm, in the form of
        {:gp-form-details gp-form-details
         :gp-hyperprior-constructor gp-hyperprior-constructor
         :hmc-options hmc-options
         :optimizer-options optimizer-options
         :debug-printouts true/false (default false)}
      which will be used to overwrite the default values."
  [X Y optimizer scaling-funcs &
   {:keys [cov-fn-form grad-cov-fn-hyper-form mean-fn-form
           gp-hyperprior-form hmc-step-size hmc-num-leapfrog-steps
           hmc-num-steps hmc-num-chains hmc-burn-in-proportion hmc-max-gps
           verbose debug-folder plot-aq]
    :or {}}]
  (let [[_ D] (mat/shape X)
        mean-fn (mean-fn-form D)
        cov-fn (cf/matern-for-vector-input D cov-fn-form)
        grad-cov-fn-hyper (cf/matern-for-vector-input D grad-cov-fn-hyper-form)
        gp-hyperprior (gp-hyperprior-form D)

        ;; Obtain particle estimate of predictive distribution on GP
        ;; hyperparameters
        alpha-particles (tufte/p :hyper-infer
                            (infer-gp-hyper X Y

                                        ;; BO options
                                        :mean-fn mean-fn
                                        :cov-fn cov-fn
                                        :grad-cov-fn-hyper grad-cov-fn-hyper
                                        :gp-hyperprior gp-hyperprior

                                        ;; HMC options
                                        :hmc-step-size hmc-step-size
                                        :hmc-num-leapfrog-steps hmc-num-leapfrog-steps
                                        :hmc-num-steps hmc-num-steps
                                        :hmc-num-chains hmc-num-chains
                                        :hmc-burn-in-proportion hmc-burn-in-proportion
                                        :hmc-max-gps hmc-max-gps))
        alphas (map first alpha-particles)
        gp-weights (map second alpha-particles)

        _ (if verbose (println :n-gps-in-acq-function (count gp-weights)))

        ;; function to create a trained-gp-obj for a given alpha currently
        ;; create-trained-gp-obj takes x and y in the form of "points" as
        ;; passed to bo-acquire, this should be changed

        ;; FIXME why have we got rid of the mean derivative but not the
        ;;
        gp-trainer (partial gp/create-trained-gp-obj
                            mean-fn cov-fn X Y)

        ;; make trained-gp-obj for each sampled alpha
        gps (mapv gp-trainer alphas)

        gp-predictors (mapv #(fn [x*] (gp/gp-predict-mu-sig % x*)) gps)

        ;; mean-best for each gp required for the respective acq-func
        all-means (mapv first (mapv #(% X) gp-predictors))
        mean-bests (mapv #(first (indexed-max identity %)) all-means)

        ;; FIXME might be better to use the single average of the means here to recover the true expected improvement?

        ;; Setup the acquistion function (will be a function of the new point
        ;; x* that returns a value and a derivative)
        xi 0 ; For now we are just going to hard-code xi to 0 for simplicity
        base-acq-func (partial acq/expected-improvement xi)
        acq-fn (partial acq/integrated-aq-func base-acq-func mean-bests gp-predictors gp-weights)
        acq-fn-single #(acq-fn [%])
        ;; Optimize the acquisition function to give the point to
        ;; evaluate next

        x-next (tufte/p :acq-opt
                        (optimizer
                          #(first (acq-fn-single %))))
        acq-opt (acq-fn-single x-next)
        _ (if verbose (println :acq-opt acq-opt))


        ;; Establish which point is best so far and the mean and std dev
        ;; for each of the evaluated points.  This is not only for sake
        ;; of the return arguments
        [means std-devs] (gp/gp-mixture-mu-sig gp-predictors gp-weights X)
        [_ i-best] (indexed-max identity means)
        ]
    (tufte/p :db-folder
     (if debug-folder
      (let [subfolder (str "bopp-debug-files/" debug-folder "/bo-step-scaled-" (System/currentTimeMillis))
            alphas-exp (mat/exp alphas)
            [_ D] (mat/shape X)
            [_ n-alpha] (mat/shape alphas-exp)
            alphas-unscaled (loop [a-un ((:log-Z-unscaler-no-centering scaling-funcs) (mat/submatrix alphas-exp 1 [0 1]))
                                   i-s 1]
                              (if (>= i-s n-alpha)
                                a-un
                                (recur (mat/join-along 1 a-un
                                                       ((:log-Z-unscaler-no-centering scaling-funcs) (mat/submatrix alphas-exp 1 [i-s 1]))
                                                       ((:theta-unscaler-no-centering scaling-funcs) (mat/submatrix alphas-exp 1 [(inc i-s) D])))
                                       (+ i-s 1 D))))
            alphas-csv (write-csv (map #(map str %) alphas-unscaled))
            gp-weights-csv (write-csv (map (comp vector str double) gp-weights))
            xs-unscaled ((:theta-unscaler scaling-funcs) X)
            ys-unscaled ((:log-Z-unscaler scaling-funcs) Y)
            xs-csv (write-csv (map #(map str %) xs-unscaled))
            ys-csv (write-csv (map (comp vector str) ys-unscaled))

            x-grid (if (<= D 2)
                     (let [start -1.49
                           end 1.49
                           step (if (= D 1) 0.002 0.02)] ; can be more general but don't want to import things since this is only temporary
                       (cartesian (repeat D (range start end step)))))
            x-grid-csv (if (<= D 2) (write-csv (map #(map str %) ((:theta-unscaler scaling-funcs) x-grid))))
            prior-mean-vals (if (<= D 2) ((:log-Z-unscaler scaling-funcs) (mean-fn (mat/matrix x-grid))))
            prior-mean-vals-csv (if (<= D 2) (write-csv (map (comp vector str) prior-mean-vals)))]
        (.mkdir (java.io.File. subfolder))
        (spit (str subfolder "/alphas.csv") alphas-csv)
        (spit (str subfolder "/gp-weights.csv") gp-weights-csv)
        (spit (str subfolder "/xs.csv") xs-csv)
        (spit (str subfolder "/ys.csv") ys-csv)
        (if (<= D 2) (spit (str subfolder "/x-grid.csv") x-grid-csv))
        (if (<= D 2) (spit (str subfolder "/prior-mean.csv") prior-mean-vals-csv))

        (if (and plot-aq (<= D 2))
          (let [acq-vals (acq-fn x-grid)
                [means std-devs] (gp/gp-mixture-mu-sig gp-predictors gp-weights x-grid)
                means-csv (write-csv (map (comp vector str) ((:log-Z-unscaler scaling-funcs) means)))
                std-devs-csv (write-csv (map (comp vector str) ((:log-Z-unscaler-no-centering scaling-funcs) std-devs)))
                acq-csv (write-csv (map (comp vector str) acq-vals))]

            (spit (str subfolder "/acq.csv") acq-csv)
            (spit (str subfolder "/mus.csv") means-csv)
            (spit (str subfolder "/sigs.csv") std-devs-csv)))))
    [x-next i-best means std-devs])))



(defn bopp-bo
  "Runs bayesian optimization.

  Accepts:
    f - target function
    aq-optimizer - acquisition function optimizer
    theta-sampler - ??

  Returns:
    Lazy list of increasingly optimal triples
    (theta, main output of f, other outputs of f)."
  [f aq-optimizer theta-sampler &
   {:keys [initial-points num-scaling-thetas num-initial-thetas cov-fn-form
           grad-cov-fn-hyper-form mean-fn-form gp-hyperprior-form
           hmc-step-size hmc-num-leapfrog-steps hmc-num-steps hmc-num-chains
           hmc-burn-in-proportion hmc-max-gps verbose debug-folder plot-aq]
    :or {;; Initialization options
         initial-points nil
         num-scaling-thetas 50
         num-initial-thetas 5

         ;; BO options
         cov-fn-form cf/matern32-plus-matern52-K
         grad-cov-fn-hyper-form cf/matern32-plus-matern52-grad-K
         mean-fn-form dp/default-mean-fn-form
         gp-hyperprior-form dp/default-double-matern-hyperprior

         ;; HMC options
         hmc-step-size 0.3
         hmc-num-leapfrog-steps 5
         hmc-num-steps 50
         hmc-num-chains 8
         hmc-burn-in-proportion 0.5
         hmc-max-gps 50

         ;; Debug options
         verbose false
         debug-folder nil
         plot-aq false}}]
  (if debug-folder
    (do
      (.mkdir (java.io.File. "bopp-debug-files"))
      (.mkdir (java.io.File. (str "bopp-debug-files/" debug-folder)))))
  (let [
        ;; Sample some thetas to use for scaling
        num-scaling-thetas (max num-initial-thetas num-scaling-thetas)
        scaling-thetas (theta-sampler num-scaling-thetas)

        ;; FIXME add code to keep randomly sampling until distinct inputs and distinct outputs have been found

        ;; Choose a subset of scaling thetas and evaluate as the starting points
        initial-thetas (mapv #(nth scaling-thetas %)
                             (take num-initial-thetas
                                   (shuffle (range 0 (count scaling-thetas)))))

        initial-points (concat initial-points
                               (map #(into []
                                           (cons % (f %))) initial-thetas))

        ;; Setup the scaling details
        theta-min (reduce clojure.core.matrix.operators/min scaling-thetas)
        theta-max (reduce clojure.core.matrix.operators/max scaling-thetas)
        initial-log-Zs (mapv second initial-points)
        log-Z-min (reduce min initial-log-Zs)
        log-Z-max (reduce max initial-log-Zs)
        scale-details-initial (sf/->scale-details-obj theta-min
                                                      theta-max
                                                      log-Z-min
                                                      log-Z-max)]
    (if verbose (do (println :initial-thetas initial-thetas)
                  (println :initial-log-Zs initial-log-Zs)))
    (letfn [(point-seq [points scale-details]
                       (lazy-seq
                        (let [_ (if verbose (println :BO-Iteration (inc (- (count points) num-initial-thetas))))
                              scaling-funcs (sf/setup-scaling-funcs
                                             scale-details)

                              theta-scaler (:theta-scaler scaling-funcs)
                              log-Z-scaler (:log-Z-scaler scaling-funcs)

                              aq-optimizer-scaled (fn [acq-fn]
                                                    (theta-scaler
                                                     (aq-optimizer
                                                      (fn [theta & args]
                                                        (apply acq-fn (theta-scaler theta) args))))) ; takes in a function to optimize

                              [theta-next-sc i-best mean-thetas-sc std-dev-thetas-sc]
                              (bo-acquire (theta-scaler
                                           (mapv first points))
                                          (log-Z-scaler
                                           (mapv second points))
                                          aq-optimizer-scaled
                                          scaling-funcs

                                          ;; TODO Make these non optional for bo-acquire

                                          ;; BO options
                                          :mean-fn-form mean-fn-form
                                          :cov-fn-form cov-fn-form
                                          :grad-cov-fn-hyper-form grad-cov-fn-hyper-form
                                          :gp-hyperprior-form gp-hyperprior-form

                                          ;; HMC options
                                          :hmc-step-size hmc-step-size
                                          :hmc-num-leapfrog-steps hmc-num-leapfrog-steps
                                          :hmc-num-steps hmc-num-steps
                                          :hmc-num-chains hmc-num-chains
                                          :hmc-burn-in-proportion hmc-burn-in-proportion
                                          :hmc-max-gps hmc-max-gps

                                          ;; Debug options
                                          :verbose verbose
                                          :debug-folder debug-folder
                                          :plot-aq plot-aq)

                              theta-next ((:theta-unscaler scaling-funcs)
                                          theta-next-sc)
                              mean-thetas ((:log-Z-unscaler scaling-funcs)
                                         mean-thetas-sc)
                              std-dev-thetas ((:log-Z-unscaler-no-centering scaling-funcs)
                                          std-dev-thetas-sc)

                              _ (if verbose (do
                                              (println :theta-best (first (nth points i-best)) "   "
                                                       :log-Z-theta-best (second (nth points i-best)) "   "
                                                       :mean-theta-best (nth mean-thetas i-best) "   "
                                                       :std-dev-theta-best (nth std-dev-thetas i-best) "   "
                                                       :i-best i-best)
                                              (println :theta-next theta-next)
                                              (println "Calling original query with theta next  ")))

                              [log-Z results] (f theta-next)
                              points (conj points
                                           [theta-next log-Z results])
                              return-val (-> (nth points (inc i-best))
                                             (assoc 1 (nth mean-thetas i-best)))

                              _ (if verbose (println :log-Z-theta-next log-Z))
                              _ (if verbose (println :log-Z-i-best (second (nth points (inc i-best)))))
                              _ (if verbose (println :theta-mean-best (take 2 return-val)))]
                          (cons return-val
                                (point-seq points
                                           (sf/update-scale-details
                                            scale-details
                                            scaling-funcs
                                            theta-next
                                            log-Z))))))]
      (point-seq initial-points scale-details-initial))))
