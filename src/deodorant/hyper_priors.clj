(ns deodorant.hyper-priors
  "Acquisition functions for Deodorant."
  (:require [deodorant.helper-functions :refer [sample* observe* normal]]))

(defn- unflatten
  "Converts a vector to nested vector for
   data of dimension dim"
   [dim v]
   (loop [vr (vec (rest v))
          vc [[(first v)]]]
      (if (= [] vr)
        vc
        (recur (vec (drop (inc dim) vr))
               (vec (concat vc [[[(first vr)] (subvec vr 1 (inc dim))]]))))))

(defn- diff-log-normpdf
  "Differential of the pdf for a 1d Gaussian"
  [m s x]
  (/ (- m x) (Math/pow s 2)))

(defn- setup-log-normal
  "Given a mean and standard deviation, returns a sampler
   probability function and grad-probability function.
   Note log-p and grad-log-p expect as inputs the log of
   the raw hyperparameters."
  [m s]
  (let [dists     (mapv (fn [a b] (normal a b))
                          m s)
        sampler   (fn [n-samples]
                    (vec (repeatedly n-samples
                                #(mapv (fn [x] (sample* x)) dists))))
        log-p     (fn [x]
                    (reduce +
                            (mapv (fn [xd dd] (observe* dd xd)) x dists)))
        grad-log-p (fn [x]
                     (mapv (fn [a b xd]
                             (diff-log-normpdf a b xd))
                        m s x))]
    [sampler log-p grad-log-p]))

(defn- log-normal-sig-f-and-rho-hyperprior
  "A gp hyperprior for distance based kernels
   such as squared exponential or Matern that apply
   a log normal to each of the hyperparameters"
  [log-sig-f-mean log-sig-f-std log-rho-mean log-rho-std]
  (let [[s-s p-s g-p-s] (setup-log-normal [log-sig-f-mean] [log-sig-f-std])
        [r-s p-r g-p-r] (setup-log-normal log-rho-mean log-rho-std)]
    {:sampler     (fn [n-samples]
                    (mapv (fn [s r] [s r]) (s-s n-samples) (r-s n-samples)))
     :log-p       (fn [alpha]
                    (+ (p-s (first alpha)) (p-r (second alpha))))
     :grad-log-p  (fn [alpha]
                    [(g-p-s (first alpha)) (g-p-r (second alpha))])}))

(defn compose-hyperpriors
  "Composes a number of hyperpriors
   to a form a single hyperprior.  Should still be used even
   if only composing a single hyperprior as adds in the
   derivative of sig-n and applies flatten / unflatten"
  [dim log-noise-mean log-noise-std & args]
  (let [unflattener     (partial unflatten dim)
        [n-s p-n g-p-n] (setup-log-normal [log-noise-mean] [log-noise-std])
        n_args          (count args)
        sampler         (fn [n-samples]
                          (map flatten (let [sig-samples    (n-s n-samples)
                                             other-samples  (mapv #((:sampler %) n-samples) args)]
                                        (mapv (fn [n] (reduce
                                                         (fn [x y] (conj x (nth y n)))
                                                        [(nth sig-samples n)] other-samples))
                                          (range n-samples)))))
        log-p          (fn [alpha]
                          (let [alpha (unflattener alpha)]
                            (+ (p-n (first alpha))
                               (reduce + (mapv (fn [a i]
                                                   ((:log-p i) a))
                                           (rest alpha) args)))))
        grad-log-p      (fn [alpha]
                          (let [alpha (unflattener alpha)]
                            (flatten (vec (concat [[(g-p-n (first alpha))]]
                                                  (mapv (fn [a i]
                                                            ((:grad-log-p i) a))
                                                    (rest alpha) args))))))]
    {:sampler sampler :log-p log-p :grad-log-p grad-log-p}))

(defn constant-length-distance-hyperprior
  "Calls log-normal-sig-f-and-rho-hyperprior when provided with
   dim as first argument and uses the same value for the rho
   details in every dimension"
   [dim log-sig-f-mean log-sig-f-std log-rho-mean log-rho-std]
  (log-normal-sig-f-and-rho-hyperprior
     log-sig-f-mean
     log-sig-f-std (vec (repeat dim log-rho-mean)) (vec (repeat dim log-rho-std))))
