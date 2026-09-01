# Post-hoc basis-control protocol

Declared after the frozen ridge screen failed H4 and before running these
additional controls.

## Why this correction is necessary

On the nominal/equality domain, `mpe_native` and `mpe_corrupt` are exactly the
same representation, yet MPE strongly beat cumulative PLE.  Geometry cannot
explain that gap.  Day 2 already showed that information-equivalent cumulative
and local PLE coordinates induce different regularized fits.  The primary
screen therefore confounds semantic geometry with basis conditioning.

## Added controls

- `ple_local`: the exact 16-dimensional local difference basis derived from
  cumulative PLE.  With an intercept, it has the same affine span.
- `ple_whitened`: cumulative PLE centered and whitened from training inputs,
  retaining 16 output coordinates and the same ridge procedure.

No primary row is overwritten.  Results go to `results/basis_controls.csv`.

## Diagnostic decisions

1. If MPE loses its nominal advantage against either information-equivalent
   control, the nominal H4 failure is explained as coordinate geometry rather
   than semantic metric evidence.
2. Native semantic evidence on cycle/tree requires MPE to beat both basis
   controls and the corrupt metric in at least 9/12 seed aggregates.
3. If MPE beats cumulative PLE but not the basis controls, the highest-potential
   claim becomes schema-stable basis selection/ensembling, not native metric
   superiority.

