# RESULTS — 8-HOUR ICLR 2027 DIRECTION SEARCH

## 1. Executive verdict

The primary screen selected **Retrieval Risk Geometry**, but the prospectively
frozen follow-up **killed its candidate-wise reliability reranker**. A final,
permanently post-hoc aggregate-risk correction then showed a real-data signal,
but failed its cross-model synthetic gate and revealed that conditional-mean
mismatch—not candidate reliability—was responsible. The Day-8 method search is
therefore closed. Prior-art risk is literal (5 = most crowded).

| Rank | Direction | ICLR potential | Novelty | Theory clarity | Empirical signal | Prior-art risk |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Retrieval Risk Geometry | 2.0 | 2.5 | 5.0 | 2.5 | 4.5 |
| 2 | General Feature Geometry | 2.0 | 1.5 | 3.0 | 2.5 | 5.0 |
| 3 | Transformer Geometry | 1.5 | 2.5 | 2.0 | 1.0 | 4.0 |
| 4 | OrbitCover Successor | 1.5 | 1.0 | 3.0 | 2.5 | 5.0 |

This is a direction-screening result, not a leaderboard claim. The eight real datasets were capped at 4,096/1,024/1,024 rows, used one frozen split and three model seeds for core cells, and used compact structurally faithful TabR/ModernNCA implementations rather than full published hyperparameter sweeps. Dataset means are the statistical units for W/L summaries; seeds quantify optimization stability and are not treated as independent datasets.

## 2. Literature subtraction

[TabR](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4ef594af0d9a519db8fb292452c461fa-Abstract-Conference.html) already establishes key-only L2 retrieval with learned keys and context-based prediction. [ModernNCA](https://openreview.net/forum?id=JytL2MrlLT) already modernizes supervised neighborhood component analysis for tabular prediction. The 2026 *Unveiling the Role of Data Uncertainty* analysis already studies when retrieval/embeddings help by uncertainty region; PLE/PLR already supply nonlinear numerical embeddings; Function Basis Encoding and 2026 learned-knot splines already learn feature geometry; AWARE already modifies retrieval with task-aligned embeddings; and Tab-PET already injects structural geometry into tabular Transformer tokens. The claim “nonlinear embeddings improve retrieval” is therefore occupied.

What remains plausibly new is narrow: an exact finite-candidate conditional prediction-risk decomposition, its local pullback interpretation, and a derived separation between **query–candidate signal compatibility** and **candidate-only reliability**. Full sources, dates, subtraction, and adjacency risks are in [LITERATURE_BOUNDARY.md](LITERATURE_BOUNDARY.md).

## 3. Retrieval Risk Law

- **A1 — proved.** Under conditional unbiasedness and covariance `Σ`, `E[(ΣᵢwᵢYᵢ-m(x))²|X]=(wᵀd)²+wᵀΣw`; independence gives `Σᵢwᵢ²σᵢ²`.
- **A2 — proved under stated constraints.** For positive-definite `H=ddᵀ+Σ`, equality-constrained optimal weights are `H⁻¹1/(1ᵀH⁻¹1)`; nonnegative weights require a convex QP. The singular pseudoinverse boundary is treated explicitly.
- **A3 — proved.** A one-neighbor rule has risk `(m(Xᵢ)-m(x))²+σᵢ²`.
- **A4 — partially proved.** First-order Taylor expansion yields `G_signal(x)=J_m(x)ᵀJ_m(x)` locally; it is not a global distance and can be degenerate.
- **A5 — proved.** Differentiable key maps induce `G_θ(x)=J_Φ(x)ᵀW_KᵀW_KJ_Φ(x)` to second order; it is a Riemannian metric only under the rank/positivity conditions in [THEORY.md](THEORY.md).

All 9/9 numerical checks passed; the largest absolute check error was 0.00189888. Exact assumptions, proofs, correlated-noise extension, singular cases, and failure boundaries are in `THEORY.md`.

## 4. Synthetic theory validation

Across 48 random A1 systems, mean absolute theory–Monte Carlo error was 0.001899. Oracle one-neighbor retrieval reduced kNN RMSE versus raw from 0.3007 to 0.2906 on S1, 0.2364 to 0.2301 on S2, 0.6194 to 0.5963 on S3, and 0.1960 to 0.1883 on S4.

The noise control is decisive mechanistically: on S3, raw-neighborhood theoretical risk was 0.3433, while the oracle risk neighborhood reached 0.0030. A signal-only oracle still had risk 0.3263; signal proximity cannot identify reliable candidates. On S4, the wrong inverse warp worsened RMSE from 0.1960 to 0.2225. On the globally linear S2, a target-guided global metric was already sufficient, as expected.

## 5. Does nonlinear geometry improve retrieval specifically?

| Prediction branch | Retrieval branch | Result |
|---|---|---|
| raw | raw | dataset-balanced Δscore 0.0000; wins/losses/ties 0/0/8; mean within-dataset seed SD 0.0000 |
| nonlinear | raw | dataset-balanced Δscore 0.0119; wins/losses/ties 7/1/0; mean within-dataset seed SD 0.0074 |
| raw | nonlinear | dataset-balanced Δscore -0.0002; wins/losses/ties 4/4/0; mean within-dataset seed SD 0.0096 |
| nonlinear | nonlinear | dataset-balanced Δscore -0.0116; wins/losses/ties 3/5/0; mean within-dataset seed SD 0.0092 |

The answer is **not cleanly**. Retrieval-only LocalWarp gave dataset-balanced Δscore -0.0002; wins/losses/ties 4/4/0; mean within-dataset seed SD 0.0096, while prediction-only LocalWarp gave dataset-balanced Δscore 0.0119; wins/losses/ties 7/1/0; mean within-dataset seed SD 0.0074 and both branches gave dataset-balanced Δscore -0.0116; wins/losses/ties 3/5/0; mean within-dataset seed SD 0.0092. Any observed gain must therefore be separated from ordinary representation effects and tested on a prospective panel.

The raw and LocalWarp branches emit the same representation dimension; LocalWarp adds only eight monotone increments per numerical feature, while branch widths and key capacity are held fixed. PLE/PLR are reported separately and are not used as capacity-matched evidence.

Per-dataset branch means, seed SDs, and deltas are in [`table_branch_ablation.csv`](table_branch_ablation.csv) and visualized in Figure 4.

## 6. TabR results

For retrieval-only LocalWarp, dataset-balanced Δscore -0.0002; wins/losses/ties 4/4/0; mean within-dataset seed SD 0.0096. The first-seed representation screen was: ple: Δ 0.0074, W/L 5/3; plr: Δ 0.0039, W/L 4/4; localwarp: Δ -0.0002, W/L 4/4. LocalWarp changed cross-fitted risk Spearman by -0.1000, selected proxy risk by +0.0492, target mismatch by +0.0481, and candidate noise by +0.0010 (negative risk components are better). It jointly improved score and reduced proxy risk on 0/8 datasets. The wrong-warp first-seed control changed score by -0.0034 on average.

Core three-seed means below are `raw → LocalWarp` (prediction-only for MLP, retrieval-only for TabR/ModernNCA). Accuracy is higher-better; standardized RMSE is lower-better.

| Dataset | Metric | MLP | TabR | ModernNCA |
|---|---|---:|---:|---:|
| adult | accuracy | 0.8460 → 0.8327 | 0.8307 → 0.8291 | 0.8460 → 0.8333 |
| churn | accuracy | 0.8662 → 0.8545 | 0.8597 → 0.8509 | 0.8669 → 0.8584 |
| higgs-small | accuracy | 0.6429 → 0.6484 | 0.6260 → 0.6315 | 0.6416 → 0.6611 |
| otto | accuracy | 0.7500 → 0.7080 | 0.7454 → 0.7500 | 0.7412 → 0.7158 |
| california | standardized_rmse | 0.4498 → 0.4764 | 0.4207 → 0.4410 | 0.4340 → 0.4570 |
| diamond | standardized_rmse | 0.2251 → 0.3295 | 0.2000 → 0.1962 | 0.1990 → 0.1993 |
| house | standardized_rmse | 0.6467 → 0.6452 | 0.6535 → 0.6162 | 0.6736 → 0.6484 |
| black-friday | standardized_rmse | 0.8742 → 0.8619 | 0.7710 → 0.7933 | 0.7332 → 0.7365 |

## 7. ModernNCA results

Retrieval-only LocalWarp transferred with dataset-balanced Δscore -0.0035; wins/losses/ties 2/6/0; mean within-dataset seed SD 0.0110. The representation screen was: ple: Δ 0.0038, W/L 2/6; plr: Δ 0.0013, W/L 3/5; localwarp: Δ -0.0035, W/L 2/6. It changed risk Spearman by -0.0163, proxy risk by +0.0105, mismatch by +0.0114, and candidate noise by -0.0009; joint score/risk improvement occurred on 1/8 datasets. Thus the risk diagnostic transfers beyond TabR, but explicit nonlinear geometry is not a uniformly beneficial method.

## 8. MLP control

Prediction-only LocalWarp produced dataset-balanced Δscore -0.0223; wins/losses/ties 3/5/0; mean within-dataset seed SD 0.0078. The first-seed representation screen was: ple: Δ 0.0187, W/L 5/3; plr: Δ 0.0248, W/L 4/3; localwarp: Δ -0.0223, W/L 3/5. This control shows that any broad LocalWarp improvement is not automatically retrieval-specific.

## 9. Learned metric field

On S1, automatic-differentiation pullbacks gave: ModernNCA/localwarp: cosine 0.693, angle 43.2°, risk ρ 0.084; ModernNCA/raw: cosine 0.798, angle 30.1°, risk ρ 0.119; TabR/localwarp: cosine 0.864, angle 19.3°, risk ρ 0.191; TabR/raw: cosine 0.887, angle 16.4°, risk ρ 0.143. Although the nonlinear map permits input dependence, the fitted LocalWarp field in Figure 3 remains nearly constant and does not recover the target field's rotations. Moreover, even high Frobenius alignment would not imply noise awareness: a symmetric query–candidate metric cannot generally represent a candidate-only heteroscedastic penalty.

## 10. Neighbor quality mechanism

Across dataset × model × representation cells, cross-fitted retrieval-risk alignment versus performance gain had Spearman `ρ=0.000`. For TabR, LocalWarp's mean proxy-risk change was +0.0492 and score change was -0.0002; for ModernNCA the corresponding values were +0.0105 and -0.0035. Therefore the answer to “do improvements retrieve lower-risk candidates?” is **mixed, not established**. This is exactly the prospective study's main falsifiable mechanism.

## 11. Key-network redundancy

Capacity interaction screen: california/deep: Δ -0.0043 ± 0.0105; california/linear: Δ -0.0017 ± 0.0022; california/shallow: Δ -0.0047 ± 0.0058; california/standard: Δ -0.0204 ± 0.0049; higgs-small/deep: Δ 0.0166 ± 0.0263; higgs-small/linear: Δ 0.0072 ± 0.0054; higgs-small/shallow: Δ 0.0130 ± 0.0082; higgs-small/standard: Δ 0.0055 ± 0.0116. Explicit LocalWarp did not show a monotone advantage as keys deepened. The strongest current interpretation is that expressive key networks can learn much of the local signal geometry, while symmetric distances still lack a direct candidate-reliability channel.

## 12. Transformer diagnostic

- S1_rotating: raw -0.3019 (first→final ρ 0.669), ple -0.2834 (first→final ρ 0.870), localwarp -0.3051 (first→final ρ 0.719); raw reference -0.3019.
- S4_warp: raw -0.2029 (first→final ρ 0.862), ple -0.1887 (first→final ρ 0.925), localwarp -0.2965 (first→final ρ 0.704); raw reference -0.2029.
- california: raw -0.4766 (first→final ρ 0.840), ple -0.4444 (first→final ρ 0.808), localwarp -0.5096 (first→final ρ 0.588); raw reference -0.4766.
- higgs-small: raw 0.6914 (first→final ρ 0.619), ple 0.6895 (first→final ρ 0.773), localwarp 0.6846 (first→final ρ 0.510); raw reference 0.6914.

First-to-final distance correlations remained between 0.510 and 0.925; geometry was transformed but not uniformly destroyed. LocalWarp did not strongly help the MLP/retrieval panel, and FT-Transformer did not specifically destroy a corresponding advantage. **DEMOTE** custom Transformer geometry; the optional intervention gate was not passed.

## 13. OrbitCover successor

No truly new theoretical extension survived. With fixed marginals, variance-optimal two-sample coupling reduces to covariance minimization, already the domain of classical and recent antithetic/optimal-coupling work. Existing OrbitCover evidence remains a valuable same-target finite-budget construction, but not a new general coupling theorem. **KEEP CURRENT PAPER SEPARATE.** See [ORBITCOVER_SUCCESSOR_AUDIT.md](ORBITCOVER_SUCCESSOR_AUDIT.md).

## 14. Failed hypotheses

- Explicit nonlinear retrieval geometry did not give a uniform real-panel benefit.
- Better exact signal-metric alignment did not reliably translate into lower candidate-noise risk.
- LocalWarp was often redundant with an expressive key encoder; the capacity interaction was not monotone.
- The raw symmetric distance cannot encode candidate-only heteroscedastic reliability in general.
- The wrong warp was not guaranteed to fail on every real dataset, even though it failed on the controlled S4 task.
- Metric-alignment/performance association was not strong enough for a causal claim.
- FT-Transformer did not uniformly destroy tokenizer geometry.
- The broad OrbitCover covariance-optimal successor was occupied by antithetic-coupling theory.

## 15. Best simple scientific insight

**A statistically good neighbor is a row with low conditional target mismatch and low candidate uncertainty—not merely a nearby row.**

## 16. Candidate ICLR thesis

The strongest supported thesis is now negative and narrower: **retrieval-based
tabular models induce local signal metrics, but mean candidate-wise risk is not
the risk of an aggregate prediction; signed mismatch cancellation and
squared-weight noise dilution must be retained.** Candidate reliability is a
valid term in the exact law but was not the source of the post-hoc real-panel
gain. This is an analysis result, not a new SOTA or ICLR-ready method thesis.

## 17. Method consequence

No new method survived the evidence gate, so no method name or victory is
claimed. The prospectively tested two-factor score
`compatibility_θ(x,i) + λ·reliability(i)` is rejected: its permutation control
was as good or better. The post-hoc full simplex QP improved real prediction,
but mismatch-only weighting matched it while reliability-only weighting was
weak or harmful. Any later work would therefore be a new conditional-mismatch
aggregation project, not confirmation of the original reliability method.

## 18. ICLR readiness

**INTERESTING NEGATIVE MECHANISM ONLY.** The decisive method pilot is now
complete and failed its frozen gates. OOF reliability lowered the selected
top-16 proxy risk on 11/12 datasets for TabR and 9/12 for ModernNCA, but score
wins were only 7/12 and 4/12, respectively. Dataset-balanced gains were smaller
than the distribution-matched permutation control for both models. The compact
models are not leaderboard configurations, but this is sufficient to stop the
specific candidate-wise reranker rather than escalating it to expensive
published implementations. The post-hoc QP subsequently passed its real-data
subgates (TabR 10/12 and ModernNCA 9/12 versus the original model) but failed
the frozen synthetic S3 transfer gate for ModernNCA. Because it was designed
after the primary failure and its effective ablation removes the claimed
reliability mechanism, it does not raise ICLR readiness.

## 19. Next 3-day experiment plan — cancelled

Freeze before outcomes these 12 untouched public datasets: bank-marketing, credit-g, electricity, jannis, covertype, and MagicTelescope for classification; abalone, cpu_act, elevators, Bike_Sharing_Demand, sulfur, and superconduct for regression. Verify availability/licensing before the freeze, then use 5 splits and 3 seeds without replacement; no dataset may overlap this panel. Compare published TabR and ModernNCA implementations, MLP, kNN, PLE, PLR, LMNN/NCA-style global metrics, raw shallow/deep keys, LocalWarp, and a single two-factor risk score. Estimate candidate reliability only from nested out-of-fold residuals/probabilities.

The originally proposed prospective panel has been run as a compact-model
pilot. Its failure means the three-day published-implementation expansion is
not justified. The single allowed post-hoc correction is also complete and its
joint stop rule failed. No larger benchmark, data-scaling sweep, or published-
implementation escalation is justified by this Day-8 evidence.

## 20. Prospective reliability follow-up

The frozen protocol, all 648 real method rows, 256 synthetic method rows,
figures, and audit are in `PROSPECTIVE_RISK_PROTOCOL.md` and
`PROSPECTIVE_RISK_RESULTS.md`.

The follow-up exposed a flaw in the original neighbor-quality diagnostic. For
normalized weights, the weighted mean of A3 one-neighbor risks exceeds the
actual A1 aggregation risk by exactly

```text
Var_w(d) + sum_i w_i(1-w_i) sigma_i^2.
```

Therefore mean top-k candidate risk discards signed-bias cancellation and the
squared-weight dilution of candidate noise. It can improve dramatically while
multi-neighbor prediction does not. On the S3 high-noise task, exact candidate
reliability reduced ModernNCA's top-16 proxy risk by 0.3155 but improved RMSE by
only 0.00065; estimated reliability did not improve RMSE. On the real panel,
risk-reduction versus score-gain Spearman correlation was 0.329 for TabR and
0.018 for ModernNCA. This is the strongest Day-8 result, but it is a negative
measurement correction rather than an ICLR-ready method.

## 21. Post-hoc aggregate-risk correction

The corrective QP optimized the actual plug-in aggregate risk over a learned-
distance shortlist. Across 12 datasets, 3 splits, 3 model seeds, and two key
models, full weighting gained `+0.01829` dataset-balanced score for TabR
(`10/12`, dataset-bootstrap 95% CI `[0.00862, 0.02846]`) and `+0.01826` for
ModernNCA (`9/12`, CI `[0.00611, 0.03268]`). It also beat the direct OOF proxy
on 9/12 and 8/12 datasets. These are meaningful post-hoc real-panel results.

They do not validate candidate reliability. Mismatch-only gains were
`+0.01878` and `+0.01802`, essentially the same as full weighting;
reliability-only gains were `-0.01183` and `+0.00099`. On S3, TabR improved
with exact and estimated QP weights on 8/8 seeds, while ModernNCA improved on
only 5/8 and estimated mean RMSE change was `+0.00003`. The frozen joint gate
therefore fails. All 216 real cells, 64 synthetic cells, solver audits, and
per-dataset results are in `POSTHOC_AGGREGATION_RESULTS.md` and
`posthoc_aggregation_audit.json`.
