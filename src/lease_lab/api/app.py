"""Thin, read-only frozen inference. No fit/simulation/evaluation imports."""

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from lease_lab.artifacts.store import (
    load_bundle,
    load_report,
    load_replays,
    load_challenger,
)
from lease_lab.artifacts.challenger import ChallengerReport
from lease_lab.config import CONFIG, ROOT
from lease_lab.contracts import DecisionRequest, DecisionResult, Metadata, TeachingCell
from lease_lab.decisions.policy import evaluate
from lease_lab.models.beta_cells import beta_summary, evidence
from lease_lab.report_contracts import EvaluationReport, ReplayReport

app = FastAPI(title="Lease Decision Lab · fictional simulation", version="1.2")


@lru_cache(maxsize=1)
def frozen():
    return load_bundle()


@app.get("/api/meta", response_model=Metadata)
def metadata() -> Metadata:
    bundle = frozen()
    return Metadata(
        provenance=bundle.provenance,
        available_methods=["beta", "laplace"] if bundle.laplace else ["beta"],
        evidence=evidence(bundle.beta),
        teaching=[
            TeachingCell(
                accepted=k,
                observed=n,
                alpha=float(k + 1),
                beta=float(n - k + 1),
                posterior=beta_summary(k + 1, n - k + 1),
            )
            for k, n in [(8, 10), (800, 1000)]
        ],
        default_request=DecisionRequest(),
        minimum_responses=CONFIG["minimum_responses"],
        stages=bundle.stages,
        case_presets=bundle.case_presets,
        limitations=[
            "All customers, responses, menus and economics are fictional.",
            "Beta intervals describe cohort means, not individual acceptance probabilities.",
            "Support is a separate heuristic; a narrow interval does not create evidence.",
            "Acceptance is treated as completed funding after fictional pre-approval.",
        ],
    )


@app.post("/api/decision", response_model=DecisionResult)
def decision(request: DecisionRequest) -> DecisionResult:
    try:
        return evaluate(request, frozen())
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/report", response_model=EvaluationReport)
def report() -> EvaluationReport:
    try:
        return load_report(frozen())
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404, detail="This offline chapter has not been built."
        ) from error


@app.get("/api/replays", response_model=ReplayReport)
def replays() -> ReplayReport:
    try:
        return load_replays(frozen())
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404, detail="Learning replays have not been built."
        ) from error


@app.get("/api/challenger", response_model=ChallengerReport)
def challenger() -> ChallengerReport:
    try:
        return load_challenger(frozen())
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail="The optional transformer benchmark has not been run for this source bundle.",
        ) from error


if (ROOT / "ui/dist").is_dir():
    app.mount("/", StaticFiles(directory=ROOT / "ui/dist", html=True), name="ui")
