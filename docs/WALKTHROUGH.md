# A one-minute review

Start with the saved screenshots and results. No setup is needed to inspect this page. Every customer, response and financial rule is fictional.

| Time | Inspect | What it shows |
| --- | --- | --- |
| 0:00–0:20 | [Growing case](../artifacts/browser-review/core-4bdfd484fa12/growing-case-full.jpg) | The mean rule prefers 10%; the cautious rule retains 9%. Mean improvement is about €60, but the model's fifth percentile (q05) is about −€27 against a €20 margin. |
| 0:20–0:35 | [December review](../artifacts/browser-review/core-4bdfd484fa12/december-review.jpg) | New prices change payment arithmetic immediately. They do not create observed response evidence, so the lab defers. |
| 0:35–0:50 | [Evidence map](../artifacts/browser-review/core-4bdfd484fa12/evidence-map.jpg) | 8/10 and 800/1000 have the same observed rate but different posterior estimates and precision. The interactive map also identifies prior-only cells; support is a separate check. |
| 0:50–1:00 | [Model audit](../artifacts/browser-review/core-4bdfd484fa12/complex-same512.jpg) | Compare point predictions under disclosed label/pretraining budgets. The [results](RESULTS.md) retain unsuccessful uncertainty experiments and limits. |

To interact, follow [setup](SETUP.md), then use the four views in the left navigation.

In **Learning replay**, select **Response mechanism changes** under **Environment**, then scroll beneath the chart. The replay starts at application **1**; use **Play**, **Step +1**, or drag the application slider to **400**. The [saved endpoint](../artifacts/browser-review/core-4bdfd484fa12/changed-response-replay.jpg) shows the controls and comparison. A pending response is not a refusal, and only chosen feedback that has arrived updates the learner. In this saved experiment, Thompson sampling loses; exploration has a cost.

## An engineer's next three stops

1. [Architecture](ARCHITECTURE.md): follow factual events into frozen JSON, then the typed API and UI. Simulation truth is kept out of learner and serving imports.
2. [Methods](METHODS.md): inspect finance conventions, the exact Beta reference, checked Laplace approximation, shared-draw improvement and separate support gate. The [specification summary](SPEC.md) records the frozen scope and data roles.
3. [Verification](VERIFICATION.md): inspect the saved run, actual export, numerical failures and targeted packaging checks. [Setup](SETUP.md) separates viewing from full benchmark reproduction.

A short introduction: “I built a lease decision lab to show when a higher expected return is enough to change an offer, and when uncertainty or missing evidence should keep the adviser from changing it. It combines a working Python/React product with reproducible synthetic experiments, including the cases where a more elaborate method did not earn its way into the decision workflow.”

Built with PriorLabs-TabPFN.
