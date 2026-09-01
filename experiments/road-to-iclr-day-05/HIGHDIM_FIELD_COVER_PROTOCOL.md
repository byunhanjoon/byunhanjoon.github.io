# High-dimensional field-wise strength-2 cover

Status: frozen before outcomes.

## Motivation and design

The primary experiments treat one joint category-map draw as one nuisance
factor. This scalability experiment instead gives every categorical field its
own binary nuisance factor: identity versus one frozen nontrivial within-field
level permutation. Together with binary target IDs, four feature orders, and
four model seeds, German Credit has 16 declared factors and 262,144 joint
cells.

Use all 32 rows of `GF(2)^5`. Two bits encode the four feature levels and two
disjoint bits encode the four seeds. Binary class/category factors use distinct
nonzero linear forms outside both two-bit subspaces. This yields pairwise
balance between every declared factor. Independently permute four-level names
and flip binary names for each randomized array.

## Experiment

- datasets: Bank Marketing (9 categorical fields), German Credit (13), and
  Compustat Korea direction (3);
- models: ordinal random forest, native CatBoost, one-hot Adam MLP;
- train/validation/test caps: 10,000/1,000/1,000;
- methods: randomized OA-32 and iid-32 over the same joint factors;
- independent design repetitions: 8; model seeds are selected by the design;
- outputs: ensemble predictions only, keeping storage compact.

Estimate between-repetition prediction risk with the unbiased sample variance
and compare Brier plus hard-label metrics. Validation/test transfer is
reported. This is a scalability demonstration; three datasets are not enough
for a new population-level performance claim.
