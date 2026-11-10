"""Offline cohort ownership: factual roles, nested subsets and one final recipe."""

import hashlib

from lease_lab.artifacts.store import digest, encode
from lease_lab.config import CONFIG
from lease_lab.data.events import Event
from lease_lab.simulation.environment import generate

FAMILIES = ("smooth", "nonlinear")


def stable_rows(rows: list[Event]) -> list[Event]:
    return sorted(rows, key=lambda e: hashlib.sha256(e.applicant_id.encode()).digest())


def ids_hash(rows: list[Event]) -> str:
    return digest(encode([e.applicant_id for e in rows]))


def data_hash(rows: list[Event]) -> str:
    return digest(encode([e.model_dump(mode="json") for e in rows]))


def offline_rows(family: str) -> list[Event]:
    return generate(CONFIG["applicants"], CONFIG["seed"], family)


def development_roles(rows: list[Event]) -> dict[str, list[Event]]:
    roles = {
        s: [e for e in rows if e.split == s]
        for s in ("fit", "diagnostic", "selection", "test")
    }
    ids = [e.applicant_id for e in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate offline applicant")
    diagnostic = stable_rows(roles["diagnostic"])
    midpoint = len(diagnostic) // 2
    roles["early_stop"], roles["report_only"] = (
        diagnostic[:midpoint],
        diagnostic[midpoint:],
    )
    return roles


def fit_subsets(fit: list[Event], sizes=(512, 1800, 3600)) -> dict[int, list[Event]]:
    ordered = stable_rows(fit)
    if max(sizes) != len(fit) or len(fit) != len({e.applicant_id for e in fit}):
        raise ValueError("nested budgets must end at the complete distinct fit cohort")
    return {n: ordered[:n] for n in sizes}


def confirmation_rows(family: str) -> list[Event]:
    namespace = CONFIG["confirmation_namespace"]
    if namespace in ("offline", "confirmation"):
        raise ValueError("comparison needs a fresh applicant namespace")
    rows = [
        e
        for e in generate(
            CONFIG["applicants"],
            CONFIG["confirmation_seed"],
            family,
            namespace=namespace,
        )
        if e.split == "test"
    ]
    if len(rows) != 900:
        raise ValueError("declared final reserve must contain 900 applicants")
    historical = {
        f"{name}-{i:06}"
        for name in ("offline", "confirmation")
        for i in range(CONFIG["applicants"])
    }
    if historical & {e.applicant_id for e in rows}:
        raise ValueError("final IDs overlap historical cohorts")
    return rows


def primary_ids(rows: list[Event], final: list[Event]) -> dict[str, list[str]]:
    result = {
        s: [e.applicant_id for e in rows if e.split == s]
        for s in ("fit", "diagnostic", "selection")
    }
    result["test"] = [e.applicant_id for e in final]
    flat = [i for values in result.values() for i in values]
    if len(flat) != len(set(flat)):
        raise ValueError("primary partitions overlap")
    return result


def role_ids(
    roles: dict[str, list[Event]], subsets: dict[int, list[Event]]
) -> dict[str, list[str]]:
    return {
        **{
            k: [e.applicant_id for e in roles[k]] for k in ("early_stop", "report_only")
        },
        **{f"fit_{n}": [e.applicant_id for e in rows] for n, rows in subsets.items()},
    }
