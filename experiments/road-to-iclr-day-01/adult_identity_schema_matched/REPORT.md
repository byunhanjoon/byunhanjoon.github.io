# Adult matched identity-versus-diversity experiment

All systems use the Day 1 TabPack split, 128-bin PLE, the tuned two-block GELU ResNet, and parameter matching. Four-member ensembles cost four fits.

| System | Accuracy | AUC | Log loss | Accuracy gain vs single PLE | AUC gain vs single PLE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Single PLE | 0.8617 | 0.9143 | 0.2997 | +0.0000 pp | +0.0000 pp |
| Single PLE + identity | 0.8693 | 0.9250 | 0.2849 | +0.7616 pp | +1.0695 pp |
| 4 PLE seeds | 0.8607 | 0.9155 | 0.2981 | -0.0983 pp | +0.1246 pp |
| 4 PLE schemas | 0.8635 | 0.9152 | 0.2982 | +0.1781 pp | +0.0976 pp |
| 4 identity seeds | 0.8724 | 0.9279 | 0.2745 | +1.0749 pp | +1.3596 pp |

## Interpretation

A single selected-identity model beats both four-fit PLE ensembles. Four schema views recover a small accuracy gain over one PLE model, but do not improve AUC as much as four ordinary seeds and remain far behind selected identity. The four-identity-seed ensemble is best, showing that representation improvement and ordinary ensemble improvement are additive rather than alternative explanations.

## Paired test-row bootstrap

Intervals below quantify uncertainty from the fixed test rows; they do not include training-protocol or dataset uncertainty.

| Comparison | Accuracy difference (95% CI) | AUC difference (95% CI) |
| --- | ---: | ---: |
| Single identity vs single PLE | +0.7616 [+0.4054, +1.0810] pp | +1.0695 [+0.8517, +1.2779] pp |
| Schema ensemble vs PLE-seed ensemble | +0.2764 [+0.0553, +0.4914] pp | -0.0270 [-0.0984, +0.0394] pp |
| Identity-seed ensemble vs PLE-seed ensemble | +1.1731 [+0.9150, +1.4188] pp | +1.2350 [+1.0666, +1.3912] pp |

## Ensemble diversity

| Family | Prediction distance | Decision disagreement |
| --- | ---: | ---: |
| 4 PLE seeds | 0.0561 | 3.97% |
| 4 PLE schemas | 0.0464 | 2.76% |
| 4 identity seeds | 0.0551 | 3.28% |

The schema ensemble keeps PLE fixed and replaces only the normalized numerical coordinates with PAD, DUPLICATE, POSNEG, or SIGNMAG. The identity model adds exact-value one-hot coordinates for the frozen Day 1 columns 3, 4, and 5.
