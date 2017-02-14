(defproject deodorant "0.1.3"
  :description "Deodorant: Solving the problems of Bayesian Optimization"
  :url "http://github.com/probprog/deodorant"
  :license {:name "GNU General Public License Version 3"
            :url "http://www.gnu.org/licenses/gpl.html"}
  :plugins [[lein-codox "0.10.2"]]
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [clojure-csv/clojure-csv "2.0.1"]
                 [clatrix "0.5.0"]
                 [org.clojure/core.memoize "0.5.8"]
                 [org.apache.commons/commons-math3 "3.6.1"]
                 [bozo/bozo "0.1.1"]
                 [colt "1.2.0"]
                 [net.mikera/core.matrix "0.49.0"]
                 [net.mikera/core.matrix.stats "0.7.0"]
                 [net.mikera/vectorz-clj "0.43.1"]
                 [org.clojure/tools.namespace "0.2.11"]
                 [com.taoensso/tufte "1.0.0-RC2"]])
