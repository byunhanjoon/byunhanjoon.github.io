# Candidate-independent stability-set coupling addendum

Status: protocol frozen after the common-coordinate stability-set gate and
before inspecting independent-candidate outcomes; analysis complete.

## Question and design

Could the 64-fit stability-set result be an artifact of using common nuisance
coordinates across candidate models? Repeat the two-independent-selector union
on all five panels, but generate four independent action blocks separately for
every candidate and method. Retain 512 draws, exact validation-winner coverage,
mean set size, and wrong-singleton probability.

## Frozen gate

Pass if the cover is at least as good as IID-U on all three metrics on at least
four of five panel means, with a strict advantage somewhere for every metric.
Held-out metrics remain descriptive.

## Outcome

The frozen gate **passes strictly on every clause in all 5/5 panels**.

- Confirmation: coverage 99.54% versus 97.66%, set size 1.033 versus 1.081,
  wrong singleton 0.46% versus 2.29%.
- External: coverage 97.36% versus 96.44%, size 1.068 versus 1.093, wrong
  singleton 2.64% versus 3.56%.
- Task-balanced: coverage 99.85% versus 98.90%, size 1.025 versus 1.087, wrong
  singleton 0.15% versus 1.10%.

External test-winner inclusion remains lower (25.0% versus 26.7%), preserving
the validation/test boundary. Common random numbers are not responsible for
the stability-set result.
