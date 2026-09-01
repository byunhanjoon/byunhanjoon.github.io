# Cross-score variance-component addendum

Status: protocol frozen after the total-variance identity gate and before
inspecting componentwise cover/IID outcomes; analysis complete.

## Question

Does strength-2 reduce cross-score variance only because its error covariance
is favorably oriented relative to the quotient residual, or does it also
reduce covariance self-interaction?

## Design

Reuse the independent A/B streams in the variance-identity audit. Store

- residual-aligned component:
  `E<r,e_A>^2 + E<r,e_B>^2 = 2<r,Cr>`;
- covariance self-interaction:
  `E<e_A,e_B>^2 = tr(C^2)`.

Compare 16-fit strength-2 and IID means within every candidate. Report
candidate wins, ratios of panel mean components, equal-source bootstrap
intervals, and the fraction of total predicted variance due to each component.

## Frozen gate

The mechanism gate passes if strength-2 has lower panel mean for
both components on at least four of five panels. This is a post-gate mechanism
analysis and cannot enlarge the cross-score selection gate.

## Outcome

The mechanism gate **passes 5/5 for both components**.

- The cover/IID residual-aligned ratios by panel range from 0.048 to 0.147.
- The cover/IID covariance-self-interaction ratios range from 0.032 to 0.140.
- All ten equal-source bootstrap intervals for the raw component difference
  exclude zero favorably.
- The residual-aligned term supplies 96.5--98.9% of pooled IID cross-score
  variance and 97.2--99.7% for the cover on four panels; it is therefore the
  dominant absolute term. The cover nonetheless reduces both terms rather
  than winning only through favorable covariance orientation.

This supplies the cleanest mechanism chain: strength balance shrinks the
prediction-error covariance, both terms in Proposition 19 fall, cross-score
RMSE falls, and validation selection stabilizes.
