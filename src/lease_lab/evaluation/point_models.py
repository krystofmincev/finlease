"""Bounded offline point fits. Inputs are factual development roles, never simulator truth."""

from dataclasses import dataclass
import time
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from lease_lab.artifacts.store import digest, encode
from lease_lab.data.events import Event, require_binary_outcomes
from lease_lab.data.features import Transform, fit_transform
from lease_lab.evaluation.metrics import losses
from lease_lab.report_contracts import (
    FeatureIdentity,
    LabelBudget,
    LabelRole,
    LearningCurvePoint,
    MethodDevelopment,
    SelectionTrial,
)

BASE_COLUMNS = [
    "intercept",
    "age_z",
    "log_turnover_z",
    "burden_z",
    "rate_z",
    "growing",
    "established",
    "restart",
]


def role(rows: list[Event]) -> LabelRole:
    return LabelRole(
        n=len(rows), ids_sha256=digest(encode([e.applicant_id for e in rows]))
    )


@dataclass
class PointDesign:
    base: Transform
    identity: FeatureIdentity

    def matrix(self, rows: list[Event], offers: list[str] | None = None) -> np.ndarray:
        X = self.base.matrix(
            [e.context for e in rows],
            offers if offers is not None else [e.offer_id for e in rows],
        )
        if self.identity.design == "without-burden":
            X = np.delete(X, 3, axis=1)
        elif self.identity.design == "rate-interactions":
            derived = X[:, 4, None] * X[:, [3, 5, 6, 7]]
            X = np.column_stack(
                [
                    X,
                    (derived - self.identity.derived_mean)
                    / self.identity.derived_scale,
                ]
            )
        return X.astype(self.identity.precision)


def design_for(
    fit: list[Event], base: Transform, design: str, precision="float64"
) -> PointDesign:
    mean, scale, columns = [], [], BASE_COLUMNS.copy()
    if design == "without-burden":
        columns.pop(3)
    elif design == "rate-interactions":
        X = base.matrix([e.context for e in fit], [e.offer_id for e in fit])
        derived = X[:, 4, None] * X[:, [3, 5, 6, 7]]
        mean, scale = derived.mean(axis=0).tolist(), derived.std(axis=0).tolist()
        if min(scale) <= 0:
            raise ValueError("constant derived feature")
        columns += [
            "rate_z*burden_z",
            "rate_z*growing",
            "rate_z*established",
            "rate_z*restart",
        ]
    elif design != "base":
        raise ValueError("unknown point feature design")
    metadata = dict(
        design=design,
        columns=columns,
        base_mean=base.mean,
        base_scale=base.scale,
        derived_mean=mean,
        derived_scale=scale,
        fit_context_n=len(fit),
        precision=precision,
    )
    return PointDesign(
        base, FeatureIdentity(**metadata, transform_sha256=digest(encode(metadata)))
    )


@dataclass
class FittedPoint:
    design: PointDesign
    estimator: LogisticRegression | HistGradientBoostingClassifier

    def predict(self, rows: list[Event], offers: list[str] | None = None) -> np.ndarray:
        return self.estimator.predict_proba(self.design.matrix(rows, offers))[:, 1]

    def all_offers(self, rows: list[Event], offer_ids: tuple[str, ...]) -> np.ndarray:
        return np.column_stack(
            [self.predict(rows, [offer] * len(rows)) for offer in offer_ids]
        )


@dataclass
class PointDevelopment:
    models: dict[str, FittedPoint]
    records: list[MethodDevelopment]
    reference: FittedPoint
    fit_count: int
    elapsed_seconds: float


class DevelopmentFailure(RuntimeError):
    def __init__(self, reason: str, trials: list[SelectionTrial], fits: int):
        super().__init__(reason)
        self.trials, self.fits = trials, fits


def choose_trial(
    trials: list[SelectionTrial], kind: str, tolerance: float
) -> SelectionTrial:
    completed = [t for t in trials if t.status == "completed"]
    if not completed:
        raise ValueError("all candidates failed")
    best = min(t.selection_log_loss for t in completed)
    tied = [t for t in completed if t.selection_log_loss <= best + tolerance]
    if kind == "logistic":
        return min(tied, key=lambda t: (t.columns, t.parameters["C"]))
    return min(tied, key=lambda t: (t.parameters["max_leaf_nodes"], t.chosen_round))


def native_stop(losses_: list[float], patience: int, tolerance: float) -> bool:
    return len(losses_) >= patience + 1 and not any(
        loss < losses_[-patience - 1] - tolerance for loss in losses_[-patience:]
    )


def tree_parameters(settings: dict, rate: float, leaves: int) -> dict:
    names = (
        "loss",
        "max_iter",
        "l2_regularization",
        "min_samples_leaf",
        "max_bins",
        "max_features",
        "categorical_features",
        "class_weight",
        "monotonic_cst",
        "interaction_cst",
        "n_iter_no_change",
        "tol",
    )
    return {
        **{k: settings["gbdt"][k] for k in names},
        "learning_rate": rate,
        "max_leaf_nodes": leaves,
        "random_state": settings["seed"],
    }


def develop_points(
    fit: list[Event],
    early_stop: list[Event],
    selection: list[Event],
    *,
    fit_subsets: dict[int, list[Event]],
    settings: dict,
) -> PointDevelopment:
    started = time.perf_counter()
    groups = [fit, early_stop, selection]
    flat = [e.applicant_id for rows in groups for e in rows]
    if len(flat) != len(set(flat)) or any(not rows for rows in groups):
        raise ValueError("distinct nonempty development roles required")
    for rows, expected in zip(groups, ["fit", "diagnostic", "selection"]):
        if any(e.split != expected or e.accepted is None for e in rows):
            raise ValueError("wrong role or unavailable response")
        require_binary_outcomes(rows)
    sizes = settings["fit_sizes"]
    if set(fit_subsets) != set(sizes) or sizes[-1] != len(fit):
        raise ValueError("declared nested fit subsets required")
    fit_by_id = {e.applicant_id: e for e in fit}
    previous = set()
    for size in sizes:
        subset = fit_subsets[size]
        current = {e.applicant_id for e in subset}
        if (
            len(current) != len(subset)
            or len(subset) != size
            or not previous <= current
            or any(fit_by_id.get(e.applicant_id) != e for e in subset)
        ):
            raise ValueError("nested subsets must contain exact factual fit rows")
        previous = current
    base = fit_transform(fit)  # full-fit, never development labels/contexts
    fit = fit_subsets[sizes[-1]]
    designs = {
        name: design_for(fit, base, name) for name in settings["logistic"]["designs"]
    }
    fixed_design = design_for(fit, base, "base", "float32")
    yv = np.array([e.accepted for e in early_stop], dtype=int)
    ys = np.array([e.accepted for e in selection], dtype=int)
    fitted, trials_by_kind, models, records = {}, {}, {}, []
    fit_count = 0

    def train(kind, design, rows, parameters, stopping=False):
        nonlocal fit_count
        if time.perf_counter() >= settings["deadline"]:
            raise TimeoutError("both-family development cutoff reached")
        before = time.perf_counter()
        estimator = (
            LogisticRegression(**parameters)
            if kind == "logistic"
            else HistGradientBoostingClassifier(
                **{k: v for k, v in parameters.items() if k != "early_stopping"},
                early_stopping=stopping,
            )
        )
        X = design.matrix(rows)
        fit_count += 1
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            if stopping:
                estimator.fit(
                    X,
                    [e.accepted for e in rows],
                    X_val=design.matrix(early_stop),
                    y_val=yv,
                )
            else:
                estimator.fit(X, [e.accepted for e in rows])
        elapsed = time.perf_counter() - before
        if time.perf_counter() >= settings["deadline"]:
            raise TimeoutError("fit crossed both-family development cutoff")
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            raise ValueError("logistic did not converge; no solver retry")
        return FittedPoint(design, estimator), elapsed

    logistic_grid = [
        (
            "logistic",
            design,
            {
                "C": C,
                "fit_intercept": False,
                "solver": settings["logistic"]["solver"],
                "max_iter": settings["logistic"]["max_iter"],
            },
        )
        for design in settings["logistic"]["designs"]
        for C in settings["logistic"]["C"]
    ]
    tree_grid = [
        ("gbdt", "base", tree_parameters(settings, rate, leaves))
        for rate in settings["gbdt"]["base_rates"]
        for leaves in settings["gbdt"]["base_leaves"]
    ]
    tree_grid += [
        (
            "gbdt",
            "without-burden",
            tree_parameters(
                settings,
                settings["gbdt"]["ablation_rate"],
                settings["gbdt"]["ablation_leaves"],
            ),
        )
    ]
    for kind, design_name, parameters in logistic_grid + tree_grid:
        candidate_id = f"{kind}-{design_name}-{len(trials_by_kind.get(kind, []))}"
        design = designs[design_name]
        trial = dict(
            fit_seconds=0.0,
            candidate_id=candidate_id,
            design=design_name,
            columns=len(design.identity.columns),
            feature_sha256=design.identity.transform_sha256,
            parameters={
                **parameters,
                **({"early_stopping": True} if kind == "gbdt" else {}),
            },
            status="completed",
            failure=None,
            selection_log_loss=None,
            selection_brier=None,
            selection_calibration_error=None,
            iterations=0,
            chosen_round=None,
            stopped_round=None,
            validation_stop=False,
            round_cap=False,
            reason="lbfgs_converged",
            train_loss=[],
            development_loss=[],
        )
        before = time.perf_counter()
        try:
            model, measured_fit = train(
                kind, design, fit, parameters, stopping=kind == "gbdt"
            )
            trial["fit_seconds"] = measured_fit
            estimator = model.estimator
            p = model.predict(selection)
            if kind == "gbdt":
                train_loss, development_loss = (
                    (-estimator.train_score_).tolist(),
                    (-estimator.validation_score_).tolist(),
                )
                chosen = int(np.argmin(development_loss[1:])) + 1
                p = next(
                    p[:, 1]
                    for i, p in enumerate(
                        estimator.staged_predict_proba(design.matrix(selection)), 1
                    )
                    if i == chosen
                )
                stopped = native_stop(
                    development_loss, parameters["n_iter_no_change"], parameters["tol"]
                )
                cap = estimator.n_iter_ == parameters["max_iter"]
                trial.update(
                    chosen_round=chosen,
                    stopped_round=estimator.n_iter_,
                    validation_stop=stopped,
                    round_cap=cap,
                    train_loss=train_loss,
                    development_loss=development_loss,
                    reason="validation_stop_and_round_cap"
                    if stopped and cap
                    else "round_cap"
                    if cap
                    else "validation_stop",
                )
            trial["iterations"] = (
                int(estimator.n_iter_[0]) if kind == "logistic" else estimator.n_iter_
            )
            brier, loss = losses(ys, p)
            ece = sum(
                abs(float(np.sum(p[mask] - ys[mask])))
                for i in range(8)
                if (
                    mask := ((p >= i / 8) & (p < (i + 1) / 8 if i < 7 else p <= 1))
                ).any()
            ) / len(ys)
            trial.update(
                selection_log_loss=loss,
                selection_brier=brier,
                selection_calibration_error=ece,
            )
            fitted[candidate_id] = model
        except (ValueError, RuntimeError, MemoryError, TimeoutError) as error:
            trial.update(
                status="budget_cutoff" if isinstance(error, TimeoutError) else "failed",
                failure=f"{type(error).__name__}: {error}",
                reason="fit_failed",
            )
        trial["elapsed_seconds"] = time.perf_counter() - before
        trials_by_kind.setdefault(kind, []).append(SelectionTrial(**trial))
    all_trials = [t for trials in trials_by_kind.values() for t in trials]
    if any(t.status == "budget_cutoff" for t in all_trials):
        raise DevelopmentFailure("incomplete bounded search", all_trials, fit_count)

    def budget(rows, kind, selected):
        empty = role([])
        stop = role(early_stop) if selected and kind == "gbdt" else empty
        select = role(selection) if selected else empty
        monitor = role(early_stop) if selected else empty
        return LabelBudget(
            fit=role(rows),
            early_stop=stop,
            selection=select,
            calibration=empty,
            monitoring=monitor,
            consumed_union_n=len(rows) + stop.n + select.n,
            observed_union_n=len(rows) + select.n + monitor.n,
            overlap=["early_stop=monitoring"] if stop.n else [],
            preprocessing_context_n=len(fit),
            support_history_n=len(fit),
            external_pretraining=kind == "tabpfn",
        )

    def finish_train(*args):
        try:
            return train(*args)
        except (ValueError, RuntimeError, MemoryError, TimeoutError) as error:
            raise DevelopmentFailure(str(error), all_trials, fit_count) from error

    for kind in ("logistic", "gbdt"):
        try:
            winner = choose_trial(
                trials_by_kind[kind], kind, settings["selection"]["tie_tolerance"]
            )
        except ValueError as error:
            raise DevelopmentFailure(str(error), all_trials, fit_count) from error
        model = fitted[winner.candidate_id]
        fit_seconds = winner.fit_seconds
        parameters = winner.parameters.copy()
        if kind == "gbdt":
            parameters["max_iter"] = winner.chosen_round
            parameters["early_stopping"] = False
            prefix = model
            model, refit_seconds = finish_train(kind, prefix.design, fit, parameters)
            fit_seconds += refit_seconds
            for rows in (fit, early_stop):
                expected = next(
                    p[:, 1]
                    for i, p in enumerate(
                        prefix.estimator.staged_predict_proba(
                            prefix.design.matrix(rows)
                        ),
                        1,
                    )
                    if i == winner.chosen_round
                )
                try:
                    np.testing.assert_allclose(
                        model.predict(rows), expected, atol=1e-10, rtol=0
                    )
                except AssertionError as error:
                    raise DevelopmentFailure(
                        "winning prefix/refit disagreement", all_trials, fit_count
                    ) from error
        name = kind + "-point-selected"
        models[name] = model
        curves = []
        for size in sizes:
            rows = fit_subsets[size]
            smaller, elapsed = (
                (model, 0.0)
                if size == len(fit)
                else finish_train(kind, model.design, rows, parameters)
            )
            iterations = (
                int(smaller.estimator.n_iter_[0])
                if kind == "logistic"
                else smaller.estimator.n_iter_
            )
            curves.append(
                LearningCurvePoint(
                    fit_n=size,
                    fit_ids_sha256=role(rows).ids_sha256,
                    train_log_loss=losses(
                        np.array([e.accepted for e in rows], dtype=int),
                        smaller.predict(rows),
                    )[1],
                    early_stop_log_loss=losses(yv, smaller.predict(early_stop))[1],
                    iterations=iterations,
                    elapsed_seconds=elapsed,
                )
            )
        records.append(
            MethodDevelopment(
                family=settings["family"],
                model=name,
                status="completed",
                selection="selected",
                feature=model.design.identity,
                fit_matrix_sha256=digest(model.design.matrix(fit).tobytes()),
                fit_data_sha256=digest(
                    encode([e.model_dump(mode="json") for e in fit])
                ),
                budget=budget(fit, kind, True),
                parameters=parameters,
                optimizer="lbfgs"
                if kind == "logistic"
                else "native stagewise gradient boosting",
                chosen_candidate=winner.candidate_id,
                trials=trials_by_kind[kind],
                curves=curves,
                fit_seconds=fit_seconds,
                prediction_seconds=0.0,
            )
        )
        fixed_parameters = (
            {"C": 1.0, "fit_intercept": False, "solver": "lbfgs", "max_iter": 200}
            if kind == "logistic"
            else {
                **tree_parameters(settings, 0.1, 7),
                "max_iter": settings["fixed_gbdt"]["max_iter"],
            }
        )
        fixed, elapsed = finish_train(
            kind, fixed_design, fit_subsets[sizes[0]], fixed_parameters
        )
        name = kind + "-point-512"
        models[name] = fixed
        records.append(
            MethodDevelopment(
                family=settings["family"],
                model=name,
                status="completed",
                selection="fixed",
                feature=fixed_design.identity,
                fit_matrix_sha256=digest(
                    fixed_design.matrix(fit_subsets[sizes[0]]).tobytes()
                ),
                fit_data_sha256=digest(
                    encode([e.model_dump(mode="json") for e in fit_subsets[sizes[0]]])
                ),
                budget=budget(fit_subsets[sizes[0]], kind, False),
                parameters=fixed_parameters,
                optimizer="lbfgs"
                if kind == "logistic"
                else "native stagewise gradient boosting",
                chosen_candidate=None,
                trials=[],
                curves=[],
                fit_seconds=elapsed,
                prediction_seconds=0.0,
            )
        )
    records.append(
        MethodDevelopment(
            family=settings["family"],
            model="tabpfn-v2-512",
            status="precommitted",
            selection="pretrained",
            feature=fixed_design.identity,
            fit_matrix_sha256=digest(
                fixed_design.matrix(fit_subsets[sizes[0]]).tobytes()
            ),
            fit_data_sha256=digest(
                encode([e.model_dump(mode="json") for e in fit_subsets[sizes[0]]])
            ),
            budget=budget(fit_subsets[sizes[0]], "tabpfn", False),
            parameters={
                "estimators": 1,
                "context_applicants": sizes[0],
                "seed": settings["seed"],
                "device": "cpu",
                "configuration_sha256": settings["tabpfn_config_sha256"],
            },
            optimizer="pretrained in-context inference; no task optimizer",
            chosen_candidate=None,
            trials=[],
            curves=[],
            fit_seconds=0.0,
            prediction_seconds=0.0,
        )
    )
    # The original full-fit C=1 base control is the same configuration as this search fit.
    reference_trial = next(
        t
        for t in trials_by_kind["logistic"]
        if t.design == "base" and t.parameters["C"] == 1.0
    )
    if reference_trial.status != "completed":
        raise DevelopmentFailure(
            "fixed full-fit reference failed", all_trials, fit_count
        )
    return PointDevelopment(
        models,
        records,
        fitted[reference_trial.candidate_id],
        fit_count,
        time.perf_counter() - started,
    )
