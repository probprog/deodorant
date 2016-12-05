(ns deodorant.scaling-functions
  "Scaling functions for Deodorant."
  (:require [clojure.core.matrix :as mat]
            [clojure.core.matrix.operators :as mop]))

(defn scale
  "Normalizes data to lie inside a hypercube bounded at [-1 1] along.

  Accepts a collection of data points [x] in which x may be a scalar or vector.

  Returns a tuple [scaled-data unscale] containing the scaled data and
  a function that inverts the transformation."
  [data]
  (let [dmax (reduce mop/max data)
        dmin (reduce mop/min data)
        dscale (mat/sub dmax dmin)
        scaler (fn [d] (mat/mul (mat/sub (mat/div (mat/sub d dmin) dscale) 0.5) 2))
        scaled (scaler data)
        unscaler (fn [d] (mat/add (mat/mul (mat/add (mat/div d 2) 0.5) dscale) dmin))
        unscaler-without-centering (fn [d] (mat/mul (mat/div d 2) dscale))]
    [scaled unscaler unscaler-without-centering scaler]))

(defn scale-points
  "Rescales points to a hypercube bounded at [-1 1].

  Accepts a collection of points [x y] in which x is a D-dimensionl
  vector and y is a scalar.

  Returns a tuple [x-scaled y-scaled u5nscale-x unscale-y] containing
  the scaled data and functions to revert the scaling."
  [points]
  (let [[X unscale-X unscale-X-no-centering] (scale (mapv first points))
        [Y unscale-Y unscale-Y-no-centering] (scale (mapv second points))]
    [X Y unscale-X unscale-Y unscale-X-no-centering unscale-Y-no-centering]))

(defrecord scale-details-obj
  [theta-min
   theta-max
   log-Z-min
   log-Z-max])

(defrecord scaling-funcs-obj
   [theta-scaler
   theta-unscaler
   theta-unscaler-no-centering
   log-Z-scaler
   log-Z-unscaler
   log-Z-unscaler-no-centering])

(defn update-scale-details
  [scale-details scaling-funcs theta-new log-Z-new]
  (if (< ((:log-Z-scaler scaling-funcs) log-Z-new) -1)
    scale-details
    (let [log-Z-max (max (:log-Z-max scale-details) log-Z-new)
          theta-min (mop/min (:theta-min scale-details) theta-new)
          theta-max (mop/max (:theta-max scale-details) theta-new)]
        (->scale-details-obj
           theta-min theta-max (:log-Z-min scale-details) log-Z-max))))

(defn setup-scaling-funcs
  "Given a scale-details-obj returns a scaling-funcs-obj"
  [scale-details]
  (let [[_ theta-unscaler theta-unscaler-no-centering theta-scaler] (scale [(:theta-min scale-details) (:theta-max scale-details)])
        [_ log-Z-unscaler log-Z-unscaler-no-centering log-Z-scaler] (scale [(:log-Z-min scale-details) (:log-Z-max scale-details)])]
    (->scaling-funcs-obj
       theta-scaler theta-unscaler theta-unscaler-no-centering log-Z-scaler log-Z-unscaler log-Z-unscaler-no-centering)))

(defn unflatten-from-sizes
  [sizes x]
  (let [sizes-this (first sizes)
        z-this (if (= (count sizes-this) 2)
                 (-> (take (reduce * sizes-this) x)
                     (mat/reshape sizes-this))
                 (if (= (first sizes-this) 1)
                   (first x)
                   (take (first sizes-this) x)))
        z-rest (if (empty? (rest sizes))
                 nil
                 (unflatten-from-sizes (rest sizes) (into [] (drop (reduce * sizes-this) x))))]
    (into [] (concat [z-this] z-rest))))

(defn flatten-unflatten
  "Returns functions for flattening and unflattening the thetas.  For example
   when sampling from a multivariate normal theta will be a nested vector
  TODO make me work for matrices"
  [x]
  (let [types (map type x)
        sizes (mapv (fn [v] (if (instance? mikera.vectorz.Vector v)
                             (mat/shape v))
                             (if (or (vector? v) (list? v) (set? v) (coll? v) (seq? v))
                                 [(count v)]
                                 [1]))
                   x)
        flatten-f (fn [y] (into [] (flatten y)))
        unflatten-f (partial unflatten-from-sizes sizes)]
    [flatten-f unflatten-f]))
