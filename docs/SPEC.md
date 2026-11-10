# Frozen scope and data roles

This is a self-contained public summary of the implemented design, not a new execution plan. The retained Python/config source and dependency locks are unchanged. The original design identity is blueprint v1.1, SHA256 `e6e8d659751eefca3b160d82e21bfb06c3fb51bd1bf153a1a82fc74f98f5cf1d`. Its original workspace instructions and private paths are not exported; this summary is not claimed to have the same byte hash.

## Decision contract

Compare a finite, fictional rate menu against a 9% incumbent. Python calculates payments, response estimates, expected contribution and paired improvement under shared parameter draws. The cautious rule ranks supported challengers by q05 of improvement and changes only above the €20 margin; ties and the €2 numerical boundary follow the retained configuration. An unusable Laplace fit falls back to exact Beta. Insufficient support or a numerically borderline cautious comparison returns review. This is uncertainty about expected contribution, not individual Bernoulli outcome noise.

Support separately checks mature response counts, observed cards/terms and fitted marginal input ranges. A narrow interval cannot create support. Exact Beta describes cohort-average probabilities under independent-cell assumptions. The pooled logistic model uses a proper-prior MAP fit and a checked Laplace approximation. Stress-check results include approximations that exceeded the declared accuracy tolerance; these checks do not certify every high-dimensional posterior.

## Synthetic worlds

`smooth-logistic-v1` retains the original smooth response/context law. `complex-economic-mixture-v1` uses correlated existing attributes and a four-state urgency/competing-offer mechanism. The evaluator exactly marginalizes the hidden states conditional on visible context. Personal hidden states, unchosen responses and future feedback never enter fitting or advice.

All mechanisms, input availability, pre-approval and economics are authored assumptions. They are not measured customer distributions. Acceptance is treated as completed funding; no separate approval/default model or individual maximum accepted rate is inferred.

## Data roles

| Role, per world | Count and use |
| --- | --- |
| Historical fit | 3,600; frozen transforms, support history and parameter fitting. Stable nested 512/1,800 subsets support controls and curves. |
| Historical diagnostic | 900; stable-ID split into 450 tree stopping/monitoring and 450 report-only observations. Original Bayesian diagnostics retain the historically exposed 900. |
| Selection | 600; factual log loss selects the declared feature/hyperparameter candidate. |
| Historical test | 900; prior implementation/repair evidence, not the new comparison's final reserve. |
| New comparison calibration | 0; the separate changed-response Platt example has its own 900-label task. |
| Final confirmation | 900; seed 20261111, namespace `comparison-confirmation-v1`, generated after both worlds' selection commitments. |

The same-512 controls share fitting IDs and base float32 matrices, plus 3,600-row preprocessing/support history. Selected logistic consumes 4,200 labels and monitors another 450 without using them for optimizer/settings selection. Selected trees consume 4,650, with stopping and monitoring referring to the same 450. External TabPFN pretraining labels/cost are not equalized. Final unsupported applicants stay in loss and policy denominators, including manual-review costs.

## Uncertainty scope

The point comparison uses logistic regression, sklearn `HistGradientBoostingClassifier` and pinned pretrained TabPFN V2. These methods report `uncertainty_method=not_estimated` and cannot enter cautious or Thompson adapters. Exact Beta remains a separate counting reference. True TabPFN fine-tuning is deferred.

A separate conditional tree-bootstrap audit held the selected recipe/transform fixed and stopped at its predeclared finite-member precision gate. It was not adopted; independent coverage and a fresh policy comparison were never opened. [Results and limits](RESULTS.md) preserve the measured no-go without calling it statistical undercoverage.

Offline fixed/greedy/Thompson replays share menus, context streams and feedback. Only chosen mature feedback updates the learner; pending is not refusal. Exploration may lose money and stationarity assumptions may fail.

The original source contains an optional check of the old sibling blueprint file if it exists. That file is not required for standalone execution. Product-local methods, configuration, schemas and saved commitments provide the exported contract. Source fingerprints cover `src/**/*.py` and top-level `configs/*.json`; run suffixes also incorporate both locks. UI, docs and tests are outside that scientific source fingerprint.

Built with PriorLabs-TabPFN.
