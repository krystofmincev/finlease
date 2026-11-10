# Verification and provenance

Qualified run: **core-4bdfd484fa12**, schema **1.2**. Python/config source SHA256: `1ec33f9b4156ba16945c6221864c66635dbf4c134a882e7b183f69cdeed7c9c7`. Selection freeze: `364e0a5f0e855d25484db854ae265f0a7a48ae4bcb1b322e53b0c57ac5b8cf28`. Public packaging preserves these identities; prose/assets are outside that scientific source fingerprint.

## Original scientific and product checks

The completed implementation recorded a 27.61-second full reproduction and 42 sklearn development fits in 0.8045 seconds. Pinned pretrained TabPFN took 47.3745 seconds in its component, 49.09 seconds command wall, with 734,838,784 peak RSS bytes. These are original host measurements, not installation-time guarantees. [Core proof](../artifacts/verification/core-4bdfd484fa12/core-proof.json) and [comparison evidence](../artifacts/verification/core-4bdfd484fa12/comparison.json) retain the data roles, measured results and pre-generation commitment.

The implementation handoff passed 72 Python tests, Ruff, five real-response Ajv contracts, TypeScript, strict completed-inventory checks and actual Chrome/API/export review. The retained [289,179-byte browser export](../artifacts/browser-review/core-4bdfd484fa12/audit-export.json) is source-identified. The accuracy checks document approximations that exceeded their declared tolerance and cases with poor interval coverage. The separate [bootstrap experiment that was not adopted](../artifacts/verification/core-4bdfd484fa12/bootstrap-audit/README.md) also remains inspectable; completion did not require a preferred model winner.

## Initial packaging checks, 10 November 2026

The following checks and the [machine-readable packaging record](evidence/packaging-checks.json) describe the completed prepublication pass. That record's `publication` value preserves the status before publication was authorized.

A new isolated environment was created on the existing macOS 15.5 arm64 host, using existing download caches. This was not a fresh-machine or cross-platform test. Python 3.12.13, uv 0.13.7, Node 24.14.0 and npm 11.9.0 were used.

- `uv sync --frozen` passed in 0.99 seconds with 31 base packages. Neither Torch nor TabPFN was installed in the saved-demo environment.
- The first server readiness check expired at 10 seconds before a listener appeared. Its log established no application cause. One explicitly reviewed retry used a 45-second bound and became ready in 1.571 seconds; both attempts and cleanup evidence are retained privately. The successful retry does not establish the cause of the first timeout.
- Actual HTTP checks passed for root HTML, both compiled assets and all five API routes. Growing + Exact Beta reproduced the €59.55 mean/−€26.96 q05 improvement, greedy 10% and cautious 9%. Invalid finance returned 422, valid recovery 200, and December required review for insufficient observed support.
- `make setup`, `npm run build --prefix ui` and `make check` passed. The build reproduced the exact retained UI bytes. Python reported 72 passed, with one installed Starlette TestClient/httpx deprecation warning; Ruff, five Ajv contracts, TypeScript and strict final comparison/serving isolation passed.
- All 155 candidate files present at the smoke check remained byte-identical after setup/build/tests; the initial 138 copied-source hashes still matched the original material. The browser export independently validated and equalled the current merged report semantically without importing Torch or TabPFN.
- Product-local documentation links, selected content/privacy checks and public-file scope were checked. No parent-workspace documentation link, machine path, targeted credential-pattern hit, symlink, environment, cache or weight file belongs to the candidate. The optional original blueprint check is disclosed in the self-contained specification.

The isolated environment was removed, the owned server/process group stopped, port 43187 verified free and the exact resource claim released. The original environment and scientific history remain local and intact. Full scientific reproduction, pretrained inference, the bootstrap rerun, new interactive browser review and mobile testing were not repeated during packaging.

## Visual and review dispositions

Screenshots are retained captures from the original product review. Some, including December, precede the final footer wording change from “Optional transformer” to “Pretrained”; they illustrate the observed behavior rather than claim exact final-UI bytes. The before-fix development screenshot is explicitly named and retained for [presentation-correction provenance](../artifacts/verification/core-4bdfd484fa12/ui-technical-correction.json). The rebuilt compiled UI matches the retained final snapshot.

Astra Max required exact pre-approval and applied-result inspection of the packaging packets. Resolved findings corrected the evidence-map screenshot wording, documented Laplace's Beta fallback, reconstructed the optional bootstrap runner's required scratch layout, and added notices for emitted Vite/Rolldown helpers. The first readiness timeout remained disclosed rather than silently retried. Reviews supplement the executed checks; they do not establish model validity.

## Final presentation review

A subsequent live presentation review inspected all four views, both model-audit budget views, December deferral and prior-only support. The walkthrough now directs readers to the replay controls below the chart and the saved endpoint at application 400. Python/config, models and scientific results remained unchanged; no experiments were rerun.

[Repository boundary and licence status](PUBLICATION.md) remain separate from these dated verification observations.

Built with PriorLabs-TabPFN.
