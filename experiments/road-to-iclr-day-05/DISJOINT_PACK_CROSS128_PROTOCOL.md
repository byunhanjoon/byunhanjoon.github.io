# Frozen protocol: unbiased cross-score from independent four-packs

Use the five frozen selection panels and 512 deterministic action draws per
dataset. At 128 fits compare:

- two independently randomized mutually-disjoint graph four-packs, crossed as
  two 64-fit quotient-prediction estimates; and
- the complete ordered-pair U-statistic over eight independently randomized
  16-fit strength-2 covers.

Both scores are exactly unbiased for quotient Brier/MSE. The independent-cover
control uses all 56 ordered block pairs, so the comparison does not advantage
packing through an arbitrary four-versus-four split. Report candidate score
RMSE and bias, panel winner agreement and validation regret, and exact closure
for products with at most 64 cells.

The frozen gate passes if pack-cross128 has lower panel-mean score RMSE on 5/5
panels and at least 20/23 full-product candidates, no lower agreement and no
higher regret on at least 4/5 panels, absolute mean bias below `1e-5` for both
methods, and numerical closure below `1e-12` wherever each four-pack exhausts
the product.
