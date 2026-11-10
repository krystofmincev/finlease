# Results and limits

Saved run **core-4bdfd484fa12**, schema **1.2**. These results concern two authored synthetic worlds, not measured lease customers. Lower log loss is better.

| Training view | Method | Smooth log loss | Complex log loss |
| --- | --- | ---: | ---: |
| Same 512 fitting labels | Logistic regression | 0.49030 | 0.56106 |
| Same 512 fitting labels | sklearn histogram gradient boosting | 0.55738 | 0.61168 |
| Same 512 fitting labels | Pretrained TabPFN V2 | 0.49576 | 0.55382 |
| Full fit + development | Selected logistic regression | 0.48813 | 0.54359 |
| Full fit + development | Selected histogram gradient boosting | 0.49607 | 0.56056 |

Each row scores the same 900 final applicants within its world: **883 supported, 17 unsupported, zero missing**. Unsupported predictions remain in losses. Policy evaluation includes all 900, including 17 manual reviews and €255 total review-handling costs. Support is an advisory heuristic, not model confidence or a correctness label.

The 512 controls share fitting IDs/base matrices and 3,600-row preprocessing/support history; external pretraining is unequalized. Selected logistic consumes 4,200 labels plus 450 monitored; selected trees consume 4,650 with overlapping stopping/monitoring. New comparison calibration labels are zero. The [data-role summary](SPEC.md#data-roles) explains the distinctions. These are not five equal-budget competitors.

Selected logistic has the lowest reported log loss among these main methods in both worlds. TabPFN improves on the same-512 logistic control in the complex world, but not the smooth world. This is a measured result under these assumptions, not a universal ranking. Inspect [raw comparison metrics](../artifacts/verification/core-4bdfd484fa12/comparison.json), the [actual development curves](../artifacts/browser-review/core-4bdfd484fa12/complex-development.jpg) and [source-identified browser export](../artifacts/browser-review/core-4bdfd484fa12/audit-export.json) for Brier scores, calibration counts, budgets and full-denominator policy value.

## Decisions and unsuccessful methods

The opening Growing + Exact Beta case reports €59.55 mean improvement for 10% over 9%, but q05 is −€26.96. The cautious rule keeps 9% against its €20 margin. These are shared-draw summaries of expected contribution, not an interval for an individual's realised profit. [Saved case checks](../artifacts/runs/core-4bdfd484fa12/case-suite.json) retain seeds/draw counts and support.

Independent Laplace reference fixtures have maximum predictive-quantile errors of 0.00418 and 0.00941 in ordinary one-/two-dimensional cases, against 0.03 tolerance. Sparse passes at 0.02973. Imbalanced, near-separated and separated fixtures fail at 0.03721, 0.17627 and 0.19443; their saved dispositions reject those approximations and return review. These checks do not certify every higher-dimensional fit. The [evaluation artifact](../artifacts/runs/core-4bdfd484fa12/evaluation.json) retains all six results, coverage counts and limits. The earlier [strict-gate replay failure](../artifacts/failures/strict-gate-replay-plateau.path-redacted.txt) is retained with local paths redacted; the same-objective numerical correction is explained in [methods](METHODS.md).

A separate fixed-recipe tree-bootstrap investigation stopped at its declared numerical gate:

| At 512 members per stream | Smooth | Complex | Allowed difference |
| --- | ---: | ---: | ---: |
| Maximum paired-improvement q05 difference | €9.05 | €15.18 | €2.00 |
| Maximum probability q05 difference | 0.01019 | 0.01485 | 0.02000 |

Status: **tail_unresolved**. Independent coverage and a fresh policy comparison were never opened. This is a finite-member precision no-go under that protocol, not evidence of statistical undercoverage or universal bootstrap invalidity. Trees remain point-only. The [retained audit](../artifacts/verification/core-4bdfd484fa12/bootstrap-audit/README.md) includes the commitment, results, prediction arrays and script; no successful experiment was substituted.

The saved changed-response replay also shows Thompson losing to the fixed policy on mean expected contribution for both uncertainty models. Pending feedback is not a refusal; only chosen, available responses update learning. The [replay artifact](../artifacts/runs/core-4bdfd484fa12/replays.json) preserves all 20-seed summaries and declared detailed paths.

All input availability, response mechanisms and economics are fictional assumptions. Exact Beta is a counting reference; point-model scores do not confer epistemic uncertainty. Fine-tuning, runtime RAG, production approval/default modelling and commercial impact are unbuilt or unmeasured. See [verification](VERIFICATION.md) for executed checks and their limits.

Built with PriorLabs-TabPFN.
