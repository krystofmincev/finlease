# Conditional tree bootstrap: numerical no-go

This separate research spike was not adopted by the product. It measures whole-applicant resampling variability conditional on the selected features, external transform and fixed 24/15 tree rounds. It is not a Bayesian posterior or model-selection uncertainty.

The predeclared two-stream 32/128/256/512-member study completed 2,050 scientific fits in 23.2356 seconds, peak RSS 209,584,128 bytes on the original macOS arm64 host. Two independent first-member parity fits are reported separately. Resamples and prediction vectors are distinct; first members reproduce exactly and incumbent self-contrast is exactly zero.

At 512 members per stream, maximum supported probability-q05 differences were 0.0101875/0.0148473, below 0.02. Paired-improvement q05 differed €9.0481/€15.1806 for smooth/complex, above €2. The audit stopped as `tail_unresolved`. Independent coverage and a fresh final comparison were never opened. No larger member count, tolerance relaxation or replacement seed was tried. This does not establish statistical undercoverage or universal bootstrap invalidity.

- [METHOD-COMMITMENT.json](METHOD-COMMITMENT.json) pins the source, selected recipe, seeds, streams, probes and limits before fitting.
- [RESULT.json](RESULT.json) preserves nested measurements; [PARITY-PROOF.json](PARITY-PROOF.json) records the independent checks.
- The four `.npy` arrays contain fitted predictions at fixed authored probes. They contain no real customers or individual hidden outcomes.
- [audit.py](audit.py), [baseline-bundle.json](baseline-bundle.json), [point-selection.json](point-selection.json) and `proposal.sha256` retain the runner and its inputs. The historical source snapshot is redundant with this package's byte-identical scientific source and is excluded. The original proposal digest remains an identity, not a claim that proposal text is included.

Opening these saved files does not run fitting. To repeat the original macOS probe study after base setup, use a new scratch directory so the retained evidence is preserved. From the package root:

```sh
audit_copy=$(mktemp -d "${TMPDIR:-/tmp}/lease-bootstrap.XXXXXX")
mkdir "$audit_copy/snapshot"
cp -R src configs "$audit_copy/snapshot/"
cp artifacts/verification/core-4bdfd484fa12/bootstrap-audit/audit.py "$audit_copy/"
cp artifacts/verification/core-4bdfd484fa12/bootstrap-audit/baseline-bundle.json "$audit_copy/"
cp artifacts/verification/core-4bdfd484fa12/bootstrap-audit/point-selection.json "$audit_copy/"
cp artifacts/verification/core-4bdfd484fa12/bootstrap-audit/proposal.sha256 "$audit_copy/"
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH="$audit_copy/snapshot/src" .venv/bin/python "$audit_copy/audit.py"
```

This optional rerun was not executed during packaging. The original script has its declared 600-second checks and recorded macOS RSS convention; it is preserved rather than presented as a newly portable runner. It does not run independent coverage, a fresh policy cohort or product adoption. Remove only your own scratch directory when finished.

Built with PriorLabs-TabPFN.
