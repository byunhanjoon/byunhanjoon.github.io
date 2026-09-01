# Literature boundary update — 2026-08-31

This update materially changes the novelty assessment from the rapid sweep.

## Identification-aware causal foundation models

The broad direction is no longer open territory. [Foundation Models for Partial Causal Identification](https://arxiv.org/abs/2608.20841), posted 2026-08-21, directly trains a causal foundation model to return bounds for partially identified intervention and counterfactual queries. Its reported binary two-variable experiments compare learned intervals with analytical bounds and report near-complete coverage. The paper explicitly frames structural assumptions as optional inputs and contrasts its full-support prior with point-estimation causal PFNs.

Several adjacent systems already cover point estimation or partial graph knowledge:

- [Do-PFN](https://arxiv.org/abs/2506.06039) pretrained PFNs on causal structures and interventions for in-context causal-effect estimation.
- [CausalPFN](https://arxiv.org/abs/2506.07918) maps raw observational datasets to treatment effects under simulated processes satisfying ignorability and reports calibrated uncertainty.
- [Use What You Know](https://arxiv.org/abs/2602.14972) conditions causal foundation models on full or partial graph information.
- [Improving TabPFN's Synthetic Data Generation by Integrating Causal Structure](https://arxiv.org/abs/2603.10254) reports that DAG-aware conditioning improves treatment-effect preservation relative to vanilla TabPFN generation.

Inference: the follow-up's predictive-versus-causal failure is a useful stress test, but it is not by itself a novel research direction. A viable contribution would need a narrower boundary not already handled above—for example relational partial identification, continuous identified sets with schema transfer, or adversarial evaluation of existing causal PFNs.

## Schema-semantic relational models

[Relational Transformer](https://openreview.net/forum?id=9yOTJfdzbs) already incorporates table and column names into cell tokenization and attributes part of its zero-shot relational transfer to schema semantics. [No Need to Train Your RDB Foundation Model](https://openreview.net/forum?id=hrtEiSftwk) likewise studies reusable column-level featurization for unseen relational databases. These works make “schema semantics help zero-shot relational prediction” too broad as a novelty claim.

Inference: the most differentiated observation in this follow-up is not that semantic metadata helps, but that simple TF-IDF is competitive on ordinary descriptions while frozen encoders separate sharply from TF-IDF under compositional negation. Because that negation condition was added as an exploratory stress test after the initial fixed-split results, it requires preregistered confirmation on a larger and less hand-authored role corpus before supporting a paper claim.
