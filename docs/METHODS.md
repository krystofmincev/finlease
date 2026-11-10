# Methods and limits

Built with PriorLabs-TabPFN.

## Factual experiment

All cases, pre-approval, terms, responses, margins and dates are authored fiction.
Acceptance means completed funding. No separate approval/default model or individual
maximum-rate threshold is estimated.

The required pretrained TabPFN V2 comparison is point-only; [its exact method, licence, measured
results and limits](TABPFN.md) are recorded separately. It does not alter the two
uncertainty models or their decision rule.

Visible context is business stage, age in years, monthly turnover in EUR cents and
pre-existing monthly debt service divided by turnover. Burden excludes the candidate
lease payment and stays fixed across offers. Historical finance is EUR31,000, 60 months,
EUR3,000 balloon in addition to the last instalment. Other terms recalculate arithmetic
but cannot support automated model advice. Input availability is assumed; turnover
and debt-service burden are authored synthetic fields, not measured customer data.
Turnover is not profit/free cash flow. Urgency, preferences and competing offers remain hidden.

Applicants are partitioned before event generation. One eligible offer is randomly
assigned and its known probability recorded. Restart historically saw 8%/9% at 1/2;
other cohorts saw all three offers at 1/3. No refusals were deleted. Fit, diagnostic
and selection counts are 3,600/900/600. Diagnostic IDs split into 450 stopping/monitoring
and 450 report-only observations. Both families save their selection commitments before
the fresh 900-row reserve (seed 20261111, namespace comparison-confirmation-v1).
Seed-20261110 confirmation results remain historical repair evidence. No fresh final
outcomes or evaluator probabilities enter fitting, transforms, settings or calibration.

The smooth family retains its original logistic response and context law. The revised
complex family shares partition/assignment rules and IDs but has correlated existing
attributes and an urgency/competing-offer response mixture. Four hidden-state weights
are conditioned on visible context and marginalized exactly; the feasible oracle uses
that observable probability, never a personal component or sampled future outcome.
These are predeclared authored hypotheses, not measured customer realism or a desired winner.

## Finance and fictional economics

Rates are nominal annual basis points, monthly r=bps/120000. For principal P, balloon B
and n months, regular unrounded payment is

    A = (P(1+r)^n - B) r / ((1+r)^n - 1)

At zero rate A=(P-B)/n. Exact rational arithmetic prevents half-cent cancellation.
Regular instalments round half up; the last settles the remaining balance and rounding
difference, with balloon in addition. Schedules requiring a negative final instalment
are rejected. Total/interest use that settled schedule; no lender APR claim is made.

Fictional accepted contributions are EUR900/1,250/1,700 at 8%/9%/10%. December's schedule
is versioned in config. Each decision costs EUR35 before the outcome. Manual review
earns no immediate income and costs EUR15. All reviews remain in economic denominators;
no later conversion is credited.

## Inference

Independent segment/offer cells update Beta(a,b) from factual acceptances/refusals.
Default Beta(1,1), weaker Beta(.5,.5), stronger Beta(2,2) are assumptions, not additional
observed applicants. Empty cells are prior-only. Intervals concern cohort-average
probability over the declared cohort context law, not personalised probabilities.
The teaching fixture's 8/10 and 800/1000 imply Beta(9,3)/Beta(801,201), means 75%/79.94%,
despite the same observed 80%. It is not a benchmark.

Logistic features are intercept, train-standardised age/log-turnover/pre-existing-burden/
rate, and three stage indicators. Proper Gaussian coefficient priors include intercept;
sigma presets are 3, 6 and 1.5. Negative log posterior is

    sum(logaddexp(0, Xβ) - y Xβ) + βᵀβ / (2 sigma²)

Gradient is Xᵀ(sigmoid(Xβ)-y)+β/sigma²; Hessian is XᵀW X+I/sigma². MAP uses analytic
derivatives and lower Cholesky L of the Hessian. Draws are mode+solve(Lᵀ,z), with
independently checked covariance orientation. One coefficient draw evaluates all rates.
No arbitrary jitter or variance clipping invents confidence.

The numerical fit gate requires infinity-norm gradient <1e-6 and conditioning <1e10.
Optimizer success is recorded separately: its tighter stopping rule can fail despite a
tiny residual. Neither flag certifies posterior accuracy.

The stricter shared prediction/sampling guard exposed one actual near-root optimizer
plateau in a delayed replay. A single Newton correction is allowed only for initial
residual in [1e-6,1e-3), usable conditioning, a strictly improved residual and no
objective increase beyond 32 machine epsilons times its scale. Curvature is recomputed
at the corrected mode; optimizer success/message and initial residual remain recorded.
The final gate is unchanged. This corrects optimization of the same objective, not
the accuracy of the Laplace approximation.

Independent 1D/2D posterior integrals compare predictive q05/q50/q95. Ordinary max errors
are about .0042/.0094 versus tolerance .03. Imbalanced, near-separated and separated
fixtures exceed it: their specific fitted approximations are rejected for sampling and
return review. Tail bounds and empirical grid refinement are recorded; refinement is
not a rigorous quantile-error bound. Low-dimensional checks do not certify all higher-
dimensional fits or provide a general per-applicant pathology detector.

## Decisions and support

Each shared draw computes U(a)=p(a)m(a)-c(a), then Δ(a)=U(a)-U(incumbent). The fictional
incumbent is 9%; unavailable/unsupported incumbents require cautious review. Self-Δ is
exactly zero. Greedy maximises mean U. Cautious ranks supported challengers by q05(Δ),
changes only above frozen EUR20 margin, otherwise retains a supported incumbent.
Ties within .01 cent use lower customer repayment then stable ID. q05 within EUR2 of
the margin returns numerical review.

Default 4,096 shared draws and seed 8128 give reproducibility. Independent-seed,
16,384-draw and independently integrated Beta-difference checks inspect precision.
Neither the seed nor the EUR2 heuristic guarantees every quantile's numerical accuracy.
P5 concerns expected contribution, not individual zero-or-profit outcome noise. It is
conditional on assumptions, not a customer-harm or repeated-selection guarantee.

Support separately requires ≥30 mature cell responses, observed card/terms and inputs
within that stage's marginal fitted ranges. An expired/new/empty action cannot be
recommended automatically even with narrow pooled intervals. These are heuristics,
not a joint-density or calibrated out-of-distribution detector. UI presets use rounded
fit-cohort medians aligned with their controls.

## Evaluation and feedback

Prediction reports Brier/log loss, eight counted calibration bins and rate/stage slices.
Primary Bayesian draws are not post-hoc recalibrated. A separate response-change
experiment fits a two-parameter Platt map on 900 mature diagnostic records and scores
900 separate records: a point comparator without calibration-uncertainty representation.

Coverage uses 30 independent datasets per probe. Beta targets independently integrated
cohort means with target Monte Carlo SE reported; Laplace targets full-visible-context
probabilities. Wilson intervals reflect finite seed count. This is fixed-world coverage,
not prior-sampled simulation-based calibration. Nonlinearity can make narrow intervals
cover badly despite useful point scores. Policy-value intervals are distinct paired
applicant bootstraps after freeze.

Fixed/greedy/Thompson share 90 initial mature responses, context streams, outcome uniforms,
menus and feedback. All may explore prior-only eligible actions in this simulation;
advisory gates stay stricter. Only chosen feedback available at the clock updates a
learner; occurrence and availability differ, pending is never refusal, duplicates fail.
Pooled models refit after 25 new mature responses. Beta counts update on arrival.

Twenty seeds × 400 decisions cover stationary, delayed, changed-response and outcome-
dependent-delay scenarios. Detailed seeds 0/1/2 were declared in advance; all 20 enter
summaries. Realised income arrives with feedback and costs occur at decision. Expected
reward/regret are separately labelled evaluator aggregates. Seed ranges are variability,
not posterior intervals. Thompson propensities are not computed. Stationary learning
can lag after change; selective delay can bias feedback. Thompson is not required to win.

Primary references: [SciPy Beta](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html),
[triangular solve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_triangular.html),
[Laplace](https://mc-stan.org/docs/reference-manual/laplace.html),
[Thompson tutorial](https://arxiv.org/abs/1707.02038). Actual installed-version tests
and saved outputs establish implementation evidence; references alone do not.

## Bounded point-model comparison

The audit adds two honest views: fixed logistic/boosted-tree/pretrained-TabPFN models using identical 512 fitting labels and base float32 matrices; and full-fit logistic/trees selected with bounded development alongside the fixed pretrained row. Exact Beta remains a separate counting reference, with original logistic/Laplace and empirical controls retained.

Logistic uses lbfgs, 3 explicit feature designs and C=.1/1/10. Trees use native stagewise gradient boosting, four base rate/leaf combinations and one without-burden ablation. Factual selection log loss chooses settings; explicit stopping data choose tree prefixes. No final-informed search, new probability calibrator, bootstrap model uncertainty or task fine-tuning is used. Native round-zero tree curves and nested label-budget curves are real saved measurements. Compare matched C or matched tree rate/leaves when inspecting feature effects.

The 512-label view shares full-fit preprocessing contexts and advisory response history; it is not a full deployment with only 512 past observations. External transformer pretraining is unequalized. Selected logistic uses 4,200 consumed labels plus 450 monitored; selected trees use 4,650, with stopping and monitoring referring to the same 450. Calibration labels are zero for this comparison; the distinct adverse Platt example still uses 900. Curves are conditional on selected settings and common preprocessing, not new generalization evidence.

All scores disclose advisory supported/unsupported and missing counts. Unsupported predictions remain scored; model confidence and experimental support are separate. Point models have no epistemic posterior, cautious P5 or Thompson adapter. Paired policy-value bootstrap intervals describe evaluator sampling variation. True TabPFN fine-tuning remains deferred.

## The revised complex world

The [complex-economic-mixture-v1 specification](SPEC.md#synthetic-worlds) replaces the former sine/quadratic stress mechanism. Existing business attributes are correlated through an authored scale law. Urgency and an attractive competing offer define four unobserved states; the evaluator integrates their conditional weights exactly. Factual responses are sampled from that visible-context marginal. No sampled personal hidden traits are retained or supplied to models or the oracle. Prices enter through actual fixed-contract monthly-payment differences in EUR10 units; an attractive alternative lowers acceptance at all offered prices.

The smooth world, its shared-draw uncertainty demonstration, replays and adverse examples are unchanged. Complex support bounds, feature transforms and the Beta cohort-target integration use the complex fit/context law. Families share IDs/partition rules, with different visible contexts and responses; methods within a family use identical final factual rows. New versions and hashes distinguish the former complex evidence from this comparison.

These are interpretable authored hypotheses, not measured realism. Business age and
input availability are assumed, segment is authored, and turnover/debt burden are
observed only within the synthetic experiment. Turnover is not profit/free cash flow.
Urgency/preferences/competing quotes remain unobserved. No approval, default,
real-customer ranking or commercial uplift claim follows.

## Conditional tree-bootstrap investigation: not adopted

The [conditional tree-bootstrap scope](SPEC.md#uncertainty-scope) was tested separately in an isolated source/config copy. Whole-applicant resamples used fixed selected features, external transform and 24/15 rounds, with each fitted member predicting all offers. This measures conditional resampling variability, not a Bayesian posterior or model-selection uncertainty ([Efron on selection-aware accuracy](https://pmc.ncbi.nlm.nih.gov/articles/PMC4207812/)).

The cost probe allowed the nested two-stream study to proceed. At 512 members, probability-q05 differences passed .02, but paired-improvement q05 differed up to EUR9.05 smooth/EUR15.18 complex against the frozen EUR2 numerical tolerance. The audit stopped as tail_unresolved; independent coverage and fresh final scoring were never opened. No tolerance/member/seed expansion followed. This is a numerical-precision no-go under that protocol, not proof of statistical undercoverage or a universal rejection of bootstrap. The [runnable evidence](../artifacts/verification/core-4bdfd484fa12/bootstrap-audit/README.md) records 2,050 scientific fits, two separate parity fits, 23.24s and 210MB RSS. The product keeps trees point-only and uses only Beta/Laplace decision adapters with the recorded numerical limits.
