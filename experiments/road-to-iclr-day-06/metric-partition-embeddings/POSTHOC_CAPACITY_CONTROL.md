# Post-hoc discrete-capacity control

Declared after inspecting the information-equivalent basis controls and before
running `u_ple`.

Local and whitened Q-PLE retain Q-PLE's affine span.  On some nominal schemas,
repeated empirical quantiles collapse stored states, so changing coordinates
cannot recover them.  MPE instead places one landmark on every nominal state.

The additional `u_ple` control uses 16 uniformly spaced PLE intervals between
the training minimum and maximum.  Stored codes are an equally spaced
permutation of the states, so on the 16-state nominal domain U-PLE can isolate
every state without native geometry.  It has the same 16-coordinate interface,
ridge selector, and splits.

Decisions:

- Nominal MPE must tie U-PLE within 0.5% mean MSE; a win would suggest another
  capacity/optimization mismatch.
- On cycle/tree, native MPE must beat U-PLE in at least 9/12 seed aggregates
  and by at least 10% mean unseen-state MSE to retain semantic evidence.

