# Method freeze

Status: **no method selected; G3 kill decision**.

E3 established nonzero oracle headroom but the sequential non-oracle gate captured only
0.27% of it in classification and 16.96% in regression on the final fresh test. The
classification fixed-mixture difference included zero, and the ordinary raw/rank
crossover required by the method thesis did not materialize. Therefore no M5 or M6
method, checkpoint, or hyperparameter set is frozen for confirmation.

`configs/e3_method_kill_calibrated.yaml` is a frozen diagnostic configuration, not a
deployable proposed method. Under the program's execution-order rule, E4/E5 and all
method-dependent real-data stages are not authorized.
