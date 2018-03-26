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
            [bozo.core :refer [lbfgs]]
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
   [hyper-prior-grad-log-p cov-fn grad-cov-fn-hyper x-diff-sq y-bar b-deterministic alpha]
  (let [[L psi]          (gp/gp-train
                           cov-fn x-diff-sq y-bar alpha)
        grad-log-lik-gp  (gp/gp-grad-log-likelihood
                           grad-cov-fn-hyper x-diff-sq alpha L psi)]
    (mat/add grad-log-lik-gp (hyper-prior-grad-log-p alpha))))

(defn- call-lbfgs-ignoring-failure
  "Calls lbfgs within a try catch and just returns start point if it fails"
  [f x0 max-iters]
  (try
    (lbfgs f x0 {:maxit max-iters})
    (catch Exception e
        x0)))

(defn- lbfgs-maximize
  "Runs gradient ascent (LBFGS) to find a local optimum.
  .  Uses bozo which unfortunately appears to not allow
   distribution (i.e. pmap) because it uses immutable objects that
   end up shared between runs.
  Accepts:
    x-start - Start points
    target-fn - Function to optimize.

  Returns:
    x* - D-dimensional vector argmax_x acq_fn(x)."
  [x-start target-fn max-iters]
  (let [;; THIS WANTS TO BE PMAP INSTEAD OF MAP BUT THIS CAUSES RANDOM FUCKUPS IN BOZO
        x-start (double-array x-start)
        f (fn [x] (let [[y gy] (target-fn (vec x))] [(double (- y)) (double-array (mat/sub gy))]))
        x-max (vec (call-lbfgs-ignoring-failure f x-start max-iters)) ; Add :iprint [1 3] to options keymap for loads of print outs
        ]
    x-max))


(defn- infer-gp-hyper
  "Takes a mean-fn, a cov-fn (with unset hyperparameters), a
   series of points and hyper-prior and returns a weighted set
   of hyperparameter samples using a HMC sampler"
  [X Y &
   {:keys [mean-fn cov-fn grad-cov-fn-hyper gp-hyperprior hmc-step-size
           hmc-num-leapfrog-steps hmc-num-mcmc-steps hmc-num-opt-steps hmc-num-chains
           hmc-burn-in-proportion hmc-max-gps b-deterministic verbose]
    :or {}}]
  (let [;; working with a zero mean GP and adding mean in later
        y-bars         (gp/subtract-mean-fn mean-fn X Y)

        ;; common term used everywhere
        x-diffs-sq     (bf/square-diff X)

        ;; function and its gradient for the log posterior of the gp hyperparameters
        f              (partial hyper-prior-log-posterior (:log-p gp-hyperprior) cov-fn x-diffs-sq y-bars)
        grad-f         (partial grad-hyper-prior-log-posterior (:grad-log-p gp-hyperprior) cov-fn grad-cov-fn-hyper x-diffs-sq y-bars b-deterministic)

        ;; hmc needs negative of this
        u              (fn [alpha] (- (f alpha)))            ; MEMOIZE ME
        grad-u         (fn [alpha] (mat/sub (grad-f alpha))) ; MEMOIZE ME

        ;; select the points at which to initialize the hmc chains
        alpha-starts   (map vec (dp/default-hmc-initializer hmc-num-chains gp-hyperprior))
        D (first (mat/shape (first alpha-starts)))
        f_grad_f (fn [x] (vec [(f x) (grad-f x)]))
        alpha-starts (map #(lbfgs-maximize % f_grad_f hmc-num-opt-steps) alpha-starts)
        ; Make sure that all alphas are within the range where they won't cause issue
        ; Can probably make this larger than 9
        alpha-starts (mapv #(mop/min (repeat D 9) (mop/max (repeat D -9) %)) alpha-starts)

        _ (if (> verbose 1)
            (println :u-starts (mapv u alpha-starts)))

        hmc-step-size-scaled (vec (repeat D hmc-step-size))
        call-hmc-sampler   (fn [q-start]
                             (take hmc-num-mcmc-steps
                                   (hmc/hmc-chain
                                    u grad-u hmc-step-size-scaled hmc-num-leapfrog-steps q-start (u q-start))))
        thin-rate 1 ; Now using max-n-gps instead
        alphas-and-us (reduce concat
                          (pmap #(hmc/burn-in-and-thin
                                    hmc-burn-in-proportion thin-rate
                                    (call-hmc-sampler %))
                                alpha-starts))
        alpha-samples (mapv first alphas-and-us)
        u-vals (mapv second alphas-and-us)
        [alpha-samples weights] (hmc/collapse-identical-samples
                                 alpha-samples
                                 verbose)
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
  sequence inputs and outputs and returns the next point x-next to
  evaluate, the index of the point from those previously evaluated
  that is considered to be the best, and the predicted mean and std-dev
  of the evaluated points.  Note this std-dev is in the estimate of the
  'true' function value at this point and does not include the noise involved
  in evaluating this function.

  Accepts:
    X - matrix of input points, first dimension represents differnt points
      second dimension the different input dimensions
    Y - a vector of associated outputs
    acq-optimizer - a function that takes in the acquistion function as
      input and returns a point in the space of x that is the estimated
      optimum of the acquisition function, subject to the constraints of
      the model.
    scaling-funcs - scaling function object ouput from sf/setup-scaling-funcs

  Options (each of these is provide as a key value pair, defaults are
               not provided as these are set by deodorant which calls this
               function which should be consulted for further info)
      cov-fn-form grad-cov-fn-hyper-form mean-fn-form
      gp-hyperprior-form hmc-step-size hmc-num-leapfrog-steps
      hmc-num-mcmc-steps hmc-num-opt-steps hmc-num-chains
      hmc-burn-in-proportion hmc-max-gps verbose debug-folder plot-aq]

  Returns:
    x-next - point that should be evaluated next (optimum of the acquistion
             function)
    i-best - index of the point expected to be most optimal under the mixture
             of GPs posterior
    means - estimated mean value for each point in X.
    std-devs - estimated standard deviation for each point in X."
  [X Y acq-optimizer scaling-funcs &
   {:keys [cov-fn-form grad-cov-fn-hyper-form mean-fn-form
           gp-hyperprior-form hmc-step-size hmc-num-leapfrog-steps
           hmc-num-mcmc-steps hmc-num-opt-steps hmc-num-chains
           hmc-burn-in-proportion hmc-max-gps verbose
           debug-folder plot-aq b-deterministic]
    :or {}}]
  (let [[_ D] (mat/shape X)
        mean-fn (mean-fn-form D)
        cov-fn (cf/matern-for-vector-input D cov-fn-form)
        grad-cov-fn-hyper (cf/matern-for-vector-input D grad-cov-fn-hyper-form)
        gp-hyperprior (gp-hyperprior-form D b-deterministic)

        ;; Obtain particle estimate of predictive distribution on GP
        ;; hyperparameters
        alpha-particles (tufte/p :hyper-infer
                            (infer-gp-hyper X Y

                                        ;; BO options
                                        :mean-fn mean-fn
                                        :cov-fn cov-fn
                                        :grad-cov-fn-hyper grad-cov-fn-hyper
                                        :gp-hyperprior gp-hyperprior
                                        :b-deterministic b-deterministic  ;; FIXME make this less of a hack

                                        ;; HMC options
                                        :hmc-step-size hmc-step-size
                                        :hmc-num-leapfrog-steps hmc-num-leapfrog-steps
                                        :hmc-num-mcmc-steps hmc-num-mcmc-steps
                                        :hmc-num-opt-steps hmc-num-opt-steps
                                        :hmc-num-chains hmc-num-chains
                                        :hmc-burn-in-proportion hmc-burn-in-proportion
                                        :hmc-max-gps hmc-max-gps

                                        :verbose verbose))
        alphas (map first alpha-particles)
        gp-weights (map second alpha-particles)

        _ (if (> verbose 1) (println :n-gps-in-acq-function (count gp-weights)))

        ;; function to create a trained-gp-obj for a given alpha currently
        ;; create-trained-gp-obj takes x and y in the form of "points" as
        ;; passed to bo-acquire, this should be changed

        gp-trainer (partial gp/create-trained-gp-obj
                            mean-fn cov-fn X Y)

        ;; make trained-gp-obj for each sampled alpha
        gps (mapv gp-trainer alphas)

        gp-predictors (mapv #(fn [x*] (gp/gp-predict-mu-sig % x*)) gps)

        ;; mean-best for each gp required for the respective acq-func
        all-means (mapv first (mapv #(% X) gp-predictors))
        mean-bests (mapv #(first (indexed-max identity %)) all-means)

        ;; Setup the acquistion function (will be a function of the new point
        ;; x* that returns a value and a derivative)
        xi 0 ; For now we are just going to hard-code xi to 0 for simplicity
        base-acq-func (partial acq/expected-improvement xi)
        acq-fn (partial acq/integrated-aq-func base-acq-func mean-bests gp-predictors gp-weights)
        acq-fn-single #(acq-fn [%])
        ;; Optimize the acquisition function to give the point to
        ;; evaluate next

        x-next (tufte/p :acq-opt
                        (acq-optimizer
                          #(first (acq-fn-single %))))
        acq-opt (acq-fn-single x-next)

        ;; Establish which point is best so far and the mean and std dev
        ;; for each of the evaluated points.  This is not only for sake
        ;; of the return arguments
        [means std-devs] (gp/gp-mixture-mu-sig gp-predictors gp-weights X)
        [_ i-best] (indexed-max identity means)]
    (if (> verbose 1) (do (println :acq-opt acq-opt)
                        (println :i-best i-best)))
    ;; If the debug folder option is set, do some extra calculations and
    ;; output all the results

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

    ;; Final return
    [x-next i-best means std-devs])))

(defn deodorant
  "Deodorant: solving the problems of Bayesian optimization.

  Deodorant is a Bayesian optimization (BO) package with three core features:
    1) Domain scaling to exploit problem independent GP hyperpriors
    2) A non-stationary mean function to allow unbounded optimization
    3) External provision of the acquisition function optimizer so that this
       can incorporate the constraints of the problem (inc equality constraints)
       and ensure that no invalid points are evaluated.

  The main intended use of the package at present is as the BO component
  for BOPP (Bayesian Optimiation for Probabilistic Programs. Rainforth T, Le TA,
  van de Meent J-W, Osborne MA, Wood F. In NIPS 2016) which provides all the
  required inputs automatically given a program.  Even when the intention is
  simply optimization, using BOPP rather than Deodorant directly is currently
  recommended.  The rational of providing Deodorant as its own independent
  package is to seperate out the parts of BOPP that are Anglican dependent and
  those that are not.  As such, one may wish to intergrate Deodorant into
  another similar package that provides all the required inputs.

  For details on the working of Deodorant, the previously referenced paper and
  its supplementary material should be consulted.

  Accepts:
    f - target function.  Takes in a single input x and returns a pair
        [f(x), other-outputs(x)].  Here other-outputs allows for additional x
        dependent variables to be returned.  For example, in BOPP then
        other-outputs(x) is a vector of program outputs from the calling the
        marginal query, with one component for each sample output from
        this marginal query.
    acq-optimizer - a function that takes in the acquistion function as
        input and returns a point in the space of x that is the estimated
        optimum of the acquisition function, subject to the constraints of
        the model.
    theta-sampler - charecterization of the input variables which can be
        sampled from to generate example inputs and initialize the scaling.
        Should be a function that takes no inputs and results valid examples
        of the input variables.  Note that the input variables are currently
        called x in the inner functions.

  Optional Inputs: (defined with key value pairs, default values shown below
                    in brackets)
    Initialization options:
      :initial-points - Pre-evaluated points (i.e. theta and output)
        in addition to those sampled by theta-sampler.  To see correct formatting,
        run without specifying and verbose > 1.
        [nil]
      :initial-thetas - Points to evaluate at start in addition to randomly sampled
        points (i.e. total number of initialization is points provided here +
        :num-initial-points). To see correct formatting, run without specifying and verbose > 1.
      :num-scaling-thetas - Number of points used to initialize scaling
        [50]
      :num-initial-points - Number of points to initialize BO
        [5]

    GP options:
      :cov-fn-form - covariance function with unset hyperparameters
        [cp/matern32-plus-matern52-K]
      :grad-cov-fn-hyper - grad of the above with respect to the hyperparameters
        [cp/matern32-plus-matern52-grad-K]
      :mean-fn-form - mean function with unset dimensionality
        [dp/default-mean-fn-form]
      :gp-hyperprior-form - constructor for the gp hyperparameter hyperprior
        [dp/default-double-matern-hyperprior]
      :b-deterministic - whether to include noise in the GP
        [false]

    HMC options:
      :hmc-step-size - HMC step size
        [0.01]
      :hmc-num-leapfrog-steps - Number of HMC leap-frog steps
        [5]
      :hmc-num-chains - Number of samplers run in parallel
        [50]
      :hmc-burn-in-proportion - Proportion of samples to throw away as burn in
        [8]
      :hmc-max-gps - Maximum number of unique GPs to keep at the end so that
                     optimization of the acqusition function does not become
                     too expensive.
        [50]
    Debug options:
      :verbose - debug print level: 0 (none) / 1 (iteration summaries) / 2 (detailed output)
        [0]
      :debug-folder - Path for the debug folder.  No output generated if path
                not  provided.  These outputs include alphas (gp hyper paramters),
                gp-weights (weights for each hyperparameter sample) etc
        [empty]
      :plot-aq - Generate debugging csv of acquisition functions
        [false]
      :invert-output-display - Displays values of (- (f theta)) instead of (f theta).
              This is because we only consider maximization such that minimization is
              done by inverting f, in which case it may be preferable to print out
              the univerted values (e.g. risk minimization in bopp).
        [false]

  Returns:
    Lazy list of increasingly optimal triples
    (theta, estimated value of (f theta) by gp, raw evaluated value of (f theta), other outputs of f)."
  [f aq-optimizer theta-sampler &
   {:keys [initial-points initial-thetas num-scaling-thetas num-initial-points cov-fn-form
           grad-cov-fn-hyper-form mean-fn-form gp-hyperprior-form b-deterministic
           hmc-step-size hmc-num-leapfrog-steps hmc-num-mcmc-steps hmc-num-opt-steps
           hmc-num-chains hmc-burn-in-proportion hmc-max-gps verbose debug-folder plot-aq invert-output-display]
    :or {;; Initialization options
         initial-points nil
         initial-thetas nil
         num-scaling-thetas 1000
         num-initial-points 5

         ;; BO options
         cov-fn-form cf/matern32-plus-matern52-K
         grad-cov-fn-hyper-form cf/matern32-plus-matern52-grad-K
         mean-fn-form dp/default-mean-fn-form
         gp-hyperprior-form dp/default-double-matern-hyperprior
         b-deterministic false

         ;; HMC options
         hmc-step-size 0.01
         hmc-num-leapfrog-steps 2; 5
         hmc-num-mcmc-steps 20; 50
         hmc-num-opt-steps 10; 15
         hmc-num-chains 4; 8
         hmc-burn-in-proportion 0.5
         hmc-max-gps 20; 50

         ;; Debug options
         verbose 0
         debug-folder nil
         plot-aq false
         invert-output-display false}}]
  (if debug-folder
    (do
      (.mkdir (java.io.File. "bopp-debug-files"))
      (.mkdir (java.io.File. (str "bopp-debug-files/" debug-folder)))))
  (let [;; Back compatibility with verbose
        verbose (or verbose 0)
        verbose (if (= true verbose) 1 verbose)

        ;; Print options at high debug level
        _ (if (> verbose 1)
            (println
             :initial-points initial-points
             :initial-thetas initial-thetas
             :num-scaling-thetas num-scaling-thetas
             :num-initial-points num-initial-points
             :mean-fn-form mean-fn-form
             :cov-fn-form cov-fn-form
             :grad-cov-fn-hyper-form grad-cov-fn-hyper-form
             :gp-hyperprior-form gp-hyperprior-form
             :b-deterministic b-deterministic
             :hmc-step-size hmc-step-size
             :hmc-num-leapfrog-steps hmc-num-leapfrog-steps
             :hmc-num-mcmc-steps hmc-num-mcmc-steps
             :hmc-num-opt-steps hmc-num-opt-steps
             :hmc-num-chains hmc-num-chains
             :hmc-burn-in-proportion hmc-burn-in-proportion
             :hmc-max-gps hmc-max-gps
             :verbose verbose
             :debug-folder debug-folder
             :plot-aq plot-aq))

        print-transform (if invert-output-display #(- %) identity)

        ;; Sample some thetas to use for scaling
        num-scaling-thetas (max num-initial-points num-scaling-thetas)
        scaling-thetas (theta-sampler num-scaling-thetas)
        [flat-f unflat-f] (sf/flatten-unflatten (first scaling-thetas))
        scaling-thetas (mapv flat-f scaling-thetas)
        
        b-integer (mapv #(or (instance? Long %) (instance? Integer %)) (first scaling-thetas))

        ;; FIXME add code to keep randomly sampling until distinct inputs and distinct outputs have been found

        ;; Choose a subset of scaling thetas and evaluate as the starting points
        initial-theta-samples (mapv #(unflat-f (nth scaling-thetas %))
                                   (take num-initial-points
                                         (shuffle (range 0 (count scaling-thetas)))))
        
        initial-thetas (concat initial-thetas initial-theta-samples)
        
        _ (if (> verbose 1)
            (println :intial-thetas initial-thetas))

        initial-points (concat initial-points
                               (map #(into []
                                           (cons % (f %)))
                                    initial-thetas))

        _ (if (> verbose 1)
            (println :intial-points initial-points))

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
    (letfn [(point-seq [points scale-details]
                       (lazy-seq
                        (let [_ (if (> verbose 0) (println "BO Iteration: " (inc (- (count points) (inc num-initial-points)))))
                              scaling-funcs (sf/setup-scaling-funcs
                                             scale-details)

                              theta-scaler (comp (:theta-scaler scaling-funcs) flat-f)
                              log-Z-scaler (:log-Z-scaler scaling-funcs)

                              aq-optimizer-scaled (fn [acq-fn]
                                                    (theta-scaler
                                                     (aq-optimizer
                                                      (fn [theta & args]
                                                        (apply acq-fn (theta-scaler theta) args))))) ; takes in a function to optimize

                              [theta-next-sc i-best mean-thetas-sc std-dev-thetas-sc]
                              (bo-acquire ((:theta-scaler scaling-funcs)
                                            (mapv flat-f
                                                  (mapv first points)))
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
                                          :b-deterministic b-deterministic

                                          ;; HMC options
                                          :hmc-step-size hmc-step-size
                                          :hmc-num-leapfrog-steps hmc-num-leapfrog-steps
                                          :hmc-num-mcmc-steps hmc-num-mcmc-steps
                                          :hmc-num-opt-steps hmc-num-opt-steps
                                          :hmc-num-chains hmc-num-chains
                                          :hmc-burn-in-proportion hmc-burn-in-proportion
                                          :hmc-max-gps hmc-max-gps

                                          ;; Debug options
                                          :verbose verbose
                                          :debug-folder debug-folder
                                          :plot-aq plot-aq)

                              theta-next ((:theta-unscaler scaling-funcs)
                                          theta-next-sc)
                              ;; Anything that is discrete we need to make sure it
                              ;; is the right type.  Note that it shoud still be integer
                              ;; valued, but it still needs its type to be changed.
                              theta-next (mapv #(if %1
                                                   (int (+ 0.49 %2))
                                                   %2)
                                            b-integer theta-next)
                              theta-next (unflat-f theta-next)
                              mean-thetas ((:log-Z-unscaler scaling-funcs)
                                         mean-thetas-sc)
                              std-dev-thetas ((:log-Z-unscaler-no-centering scaling-funcs)
                                          std-dev-thetas-sc)

                              _ (if (> verbose 0)
                                    (do (println "Theta to evaluate next: " theta-next)))

                              [log-Z results] (f theta-next)
                              points (conj points
                                           [theta-next log-Z results])
                              best-point (nth points (inc i-best))
                              return-val [(first best-point)
                                          (nth mean-thetas i-best)
                                          (second best-point)
                                          (last best-point)]

                              ;;_ (if verbose (println :log-Z-i-best (second (nth points (inc i-best)))))
                              _ (if (> verbose 0)
                                        (do (println "Best theta: " (first return-val))
                                            (println "GP mixture estimate of (f best-theta): " (print-transform (second return-val)))
                                            (println "Evaluated (f best-theta): " (print-transform (nth return-val 2)))))
                              _ (if (> verbose 0) (println "Function value at theta next: " (print-transform log-Z) "\n"))]
                          (cons return-val
                                (point-seq points
                                           (sf/update-scale-details
                                            scale-details
                                            scaling-funcs
                                            (flat-f theta-next)
                                            log-Z))))))]
      (point-seq initial-points scale-details-initial))))
