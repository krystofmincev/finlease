"""Offline orchestration only. Serving never imports this module."""

import importlib.metadata
import json
import resource
import platform
import time

from lease_lab.artifacts.store import (
    FrozenBundle,
    digest,
    publish_bundle,
    write_json,
)
from lease_lab.config import CONFIG, ROOT
from lease_lab.evaluation.cohorts import offline_rows
from lease_lab.report_contracts import SelectionCheckpoint


from lease_lab.builds.pooled import build_pooled


def build_learning() -> FrozenBundle:
    from lease_lab.artifacts.store import load_report
    from lease_lab.evaluation.adverse import adverse_checks
    from lease_lab.evaluation.replays import build_replays

    started = time.perf_counter()
    base = build_pooled(activate=False)
    pooled_directory = ROOT / "artifacts/runs" / base.provenance.run_id
    selection_raw = (pooled_directory / "selection.json").read_bytes()
    selection = SelectionCheckpoint.model_validate_json(selection_raw)
    split_export = json.loads((pooled_directory / "split-ids.json").read_bytes())
    report = load_report(base)
    if not all(
        c.status == "pass" for c in report.numerical if c.category == "ordinary"
    ):
        raise ValueError("pooled replay requires the ordinary numerical-reference gate")
    provenance = base.provenance.model_copy(
        update={"run_id": "core-" + base.provenance.run_id.split("-", 1)[1]}
    )
    base = base.model_copy(
        update={
            "provenance": provenance,
            "stages": base.stages
            + [
                "3: exact and pooled matched-feedback replay",
                "4: adverse scenarios and evidence export; independent review recorded in docs/reviews",
            ],
        }
    )
    replay_report = build_replays(provenance)
    findings, prediction = adverse_checks(offline_rows("smooth"), base)
    report = report.model_copy(
        update={
            "provenance": provenance,
            "findings": report.findings + findings,
            "prediction": report.prediction + prediction,
            "elapsed_seconds": time.perf_counter() - started,
            "commands": [
                "uv run lease-lab build --through core",
                "uv run pytest",
                "npm run build --prefix ui",
            ],
        }
    )
    directory = publish_bundle(base, activate=False)
    write_json(directory / "evaluation.json", report)
    write_json(
        directory / "selection.json",
        selection.model_copy(
            update={
                "provenance": provenance,
                "pre_generation_sha256": digest(selection_raw),
            }
        ),
    )
    # Compact, fully typed JSON keeps genuine detailed replay arrays bounded.
    write_json(directory / "replays.json", replay_report, compact=True)
    write_json(
        directory / "manifest.json",
        {
            "provenance": provenance.model_dump(),
            "command": "uv run lease-lab build --through core",
            "elapsed_seconds": report.elapsed_seconds,
            "replay_elapsed_seconds": replay_report.elapsed_seconds,
            "settings": CONFIG,
            "comparison_config_sha256": report.comparison.config_sha256,
            "selection_freeze_sha256": report.comparison.freeze_sha256,
            "pre_generation_checkpoint_sha256": digest(selection_raw),
            "development_fit_count": report.comparison.development_fit_count,
            "development_seconds": report.comparison.development_seconds,
            "extra_reference_fits": 0,
            "reused_reference_configurations": 2,
            "separate_platt_calibrator_fits": 1,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "runtime": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": {
                p: importlib.metadata.version(p)
                for p in (
                    "numpy",
                    "scipy",
                    "scikit-learn",
                    "fastapi",
                    "pydantic",
                    "uvicorn",
                )
            },
            "python_lock_sha256": digest((ROOT / "uv.lock").read_bytes()),
            "frontend_lock_sha256": digest(
                (ROOT / "ui/package-lock.json").read_bytes()
            ),
            "split_counts": {
                name: getattr(report.datasets[0], name)
                for name in ("fit", "diagnostic", "selection", "test")
            },
            "code_tracking": "content-addressed source; no Git repository initialised",
            "seeds": {
                "offline": CONFIG["seed"],
                "coverage": list(range(9000, 9000 + CONFIG["coverage_seeds"])),
                "replay_stream_roots": list(
                    range(60000, 60000 + CONFIG["replay_seeds"])
                ),
                "replay_initial": list(range(50000, 50000 + CONFIG["replay_seeds"])),
                "detailed_paths": [0, 1, 2],
            },
        },
    )
    write_json(
        directory / "split-ids.json",
        {**split_export, "provenance": provenance.model_dump(mode="json")},
    )
    from lease_lab.contracts import DecisionRequest
    from lease_lab.decisions.policy import evaluate

    cases = [
        evaluate(
            DecisionRequest(context=c, method=m, draw_count=n, draw_seed=s), base
        ).model_dump(mode="json")
        for c in base.case_presets
        for m in ("beta", "laplace")
        for n, s in [(4096, 8128), (4096, 919), (4096, 1234), (16384, 8128)]
    ]
    write_json(
        directory / "case-suite.json",
        {
            "provenance": provenance.model_dump(),
            "cases": cases,
            "selection": "All four predeclared segment medians; no winner-based case search.",
        },
    )
    publish_bundle(base)
    return base
