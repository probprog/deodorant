(ns deodorant.broadcast-functions
  (:require [clojure.core.matrix :refer [matrix shape transpose add sub div broadcast slice-views pow]]))

(defn square-diff
  "Calculates array of squared differences. Accepts an [N D] array of
  points xs. Returns an [N D N] array in which entry [i d j] is given
  by (xs[i,d] - xs[j,d])^2."
  [xs]
  (let [xm (matrix xs)
        [N D] (shape xm)
        xx (broadcast (transpose xm) [N D N])]
    (pow (sub xx (transpose xx)) 2)))

(defn broadcast-function-NxD-MxD
  "Performs an operation between a NxD array
   and a MxD array, returning a NxDxM array.
   Note if both inputs are NxD then the output
   will be NxDxN.

   Accepts: op  - matrix operation to perform (must be from core.matrix)
            xs  - NxD  array
            zs  - NxM  array
            bFlip - if true then NN is treated as
                    first input in op
  Retuns: a NxDxM array"
  [op xs zs & b-Flip]
  (let [xm (matrix xs)
        zm (matrix zs)
        [N D] (shape xm)
        [M D_z] (shape zm)
        _ (assert (= D D_z) "Second dimensions of xs and zs must be equal")
        x-plate (transpose xm)
        z-plate (transpose zm)
        x-volume (transpose (broadcast x-plate [M D N]))
        z-volume (broadcast z-plate [N D M])]
    (if b-Flip
      (op z-volume x-volume)
      (op x-volume z-volume))))

(defn scale-square-diff
  "Scales the output from square-diff. Care should be taken
   to ensure inputs are of matrix type as will be slow
   otherwise.

   Accepts: x-sq-diff - NxDxM array to scale
            rho - vector of length scales

  Returns: the scaled version of x-sq-diff"
  [x-sq-diff rho]
  (let [[N D M] (shape x-sq-diff)
        rrho (-> (broadcast rho [M D])
                 transpose
                 (broadcast [N D M]))]
    (div x-sq-diff rrho)))

(defn safe-sum
  "Sums out a dimension of nd-array.  Takes an array
   of matrix type and a dimension to sum out. Ensures
   no crash if X is nil.

  Accepts: X - matrix
         dim - dimension to remove.

  Returns: X with dim summed out"
  [X dim]
  (if (or (= nil (first X)) (> dim (dec (count (shape X)))))
    X
    (reduce add (slice-views X dim))))

(defn broadcast-function-NxDxN-NxN
  "Performs an operation between a NxDxN array
   and a NxN array.

   Accepts: op  - matrix operation to perform (must be from core.matrix)
            NDN - NxDxN array
            NN  - NxN   array
            bFlip - if true then NN is treated as
                    first input in op
  Retuns: a NxDxN array"
  [op NDN NN & b-Flip]
  (let [[N D _] (shape NDN)
        NNb     (slice-views (broadcast NN [D N N]) 1)]
    (if b-Flip
      (op NNb NDN)
      (op NDN NNb))))

(defn safe-broadcast-op
  "Calls broadcast-function-NxD-MxD when given arrays and
  just does standard broadcasting if one is a vector.
  See broadcast-function-NxD-MxD for details on call
  structure."
  [op M1 M2 & b-Flip]
  (let [[_ D1] (shape M1)
        [_ D2] (shape M2)]
    (if (or (= D1 nil) (= D2 nil))
      (let [out (if b-Flip
                  (op M2 M1)
                  (op M1 M2))
            [No Do] (shape out)]
        (if (= Do nil)
          (matrix [out])
          (matrix out)))
      (apply broadcast-function-NxD-MxD op M1 M2 b-Flip))))
