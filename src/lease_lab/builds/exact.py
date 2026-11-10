"""Offline orchestration only. Serving never imports this module."""

import importlib.metadata
import platform
import time
import numpy as np

from lease_lab.artifacts.store import (
    FeatureBounds,
    FrozenBundle,
    Range,
    SupportProfile,
    digest,
    encode,
    publish_bundle,
    source_digest,
    write_json,
)
from lease_lab.config import CONFIG, CONFIG_PATH, ROOT, SEGMENTS
from lease_lab.contracts import Context, Provenance
from lease_lab.data.events import require_binary_outcomes
from lease_lab.models.beta_cells import fit_beta
from lease_lab.simulation.environment import generate


def support_profile(fit) -> SupportProfile:
    """One fit-only context/support identity builder for each authored family."""
    if not fit or any(e.split != "fit" for e in fit):
        raise ValueError("support profile requires factual fit rows")

    def span(name, events=fit):
        values = [getattr(e.context, name) for e in events]
        return Range(minimum=float(min(values)), maximum=float(max(values)))

    per_segment = {
        s: FeatureBounds(
            **{
                name: span(name, [e for e in fit if e.context.segment == s])
                for name in (
                    "business_age_years",
                    "monthly_turnover_cents",
                    "existing_debt_burden",
                )
            }
        )
        for s in SEGMENTS
    }
    return SupportProfile(
        business_age_years=span("business_age_years"),
        monthly_turnover_cents=span("monthly_turnover_cents"),
        existing_debt_burden=span("existing_debt_burden"),
        observed_card_ids=sorted({e.card_id for e in fit}),
        observed_policy_versions=sorted({e.policy_version for e in fit}),
        observed_rate_bps=sorted({int(e.offer_id[1:]) for e in fit}),
        by_segment=per_segment,
    )


def build_exact(activate: bool = True) -> FrozenBundle:
    started = time.perf_counter()
    rows = generate(CONFIG["applicants"], CONFIG["seed"])
    fit = [e for e in rows if e.split == "fit"]
    require_binary_outcomes(fit)
    data = encode([e.model_dump(mode="json") for e in rows])
    config_hash, source_hash = digest(CONFIG_PATH.read_bytes()), source_digest()
    blueprint = ROOT.parent / "research/lease-decision-lab-2026-11-09/BLUEPRINT.md"
    if (
        blueprint.exists()
        and digest(blueprint.read_bytes()) != CONFIG["blueprint_sha256"]
    ):
        raise ValueError(
            "selected blueprint changed; reconcile its identity before rebuilding"
        )
    dependency_hash = digest(
        (ROOT / "uv.lock").read_bytes() + (ROOT / "ui/package-lock.json").read_bytes()
    )
    provenance = Provenance(
        schema_version="1.2",
        origin="authored-fictional-simulation",
        simulation_only=True,
        blueprint_version="1.1",
        currency="EUR",
        money_unit="cent",
        policy_version="fictional-policy-1.0",
        blueprint_sha256=CONFIG["blueprint_sha256"],
        config_sha256=config_hash,
        source_sha256=source_hash,
        data_sha256=digest(data),
        run_id="exact-"
        + digest(data + source_hash.encode() + dependency_hash.encode())[:12],
        seed=CONFIG["seed"],
    )

    presets = []
    for segment in SEGMENTS:
        cohort = [e.context for e in fit if e.context.segment == segment]
        presets.append(
            Context(
                segment=segment,
                business_age_years=round(
                    float(np.median([c.business_age_years for c in cohort])) * 2
                )
                / 2,
                monthly_turnover_cents=round(
                    float(np.median([c.monthly_turnover_cents for c in cohort]))
                    / 100000
                )
                * 100000,
                existing_debt_burden=round(
                    float(np.median([c.existing_debt_burden for c in cohort])), 2
                ),
            )
        )
    bundle = FrozenBundle(
        provenance=provenance,
        beta=fit_beta(fit),
        support=support_profile(fit),
        case_presets=presets,
        stages=[
            "0: contracts and first offline artifact",
            "1: exact cockpit; focused checks and actual browser review",
        ],
    )
    directory = publish_bundle(bundle, activate=activate)
    write_json(
        ROOT / ".local/data" / (provenance.run_id + ".json"),
        [e.model_dump(mode="json") for e in rows],
    )
    split_ids = {
        name: [e.applicant_id for e in rows if e.split == name]
        for name in ("fit", "diagnostic", "selection", "test")
    }
    write_json(
        directory / "split-ids.json",
        {"provenance": provenance.model_dump(), "ids": split_ids},
    )
    write_json(
        directory / "manifest.json",
        {
            "provenance": provenance.model_dump(),
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
            "split_counts": {k: len(v) for k, v in split_ids.items()},
            "fit_accepted": sum(e.accepted is True for e in fit),
            "fit_refused": sum(e.accepted is False for e in fit),
            "command": "uv run lease-lab build --through exact",
            "elapsed_seconds": time.perf_counter() - started,
            "workers": 1,
            "randomness": {
                "root": CONFIG["seed"],
                "streams": ["split", "context", "assignment", "response"],
            },
        },
    )
    return bundle
