"""LightGBM models for each stat's mean, plus the pieces that make them probabilistic."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from .distributions import Dispersion, Recalibration, StatModel
from .features import FEATURE_GROUPS, FEATURES

log = logging.getLogger(__name__)

BASE_PARAMS: dict[str, Any] = {
    "objective": "poisson",  # counts with a log link: predictions stay positive, variance grows with the mean
    "metric": "poisson",
    "verbosity": -1,
    "bagging_freq": 1,
    "num_threads": 0,
}
DEFAULT_PARAMS: dict[str, Any] = {
    "learning_rate": 0.04,
    "num_leaves": 31,
    "min_data_in_leaf": 80,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "lambda_l2": 1.0,
}
MAX_ROUNDS = 3000
EARLY_STOP = 100


def _matrix(X: pd.DataFrame) -> pd.DataFrame:
    return X[FEATURES].astype(float)


def fit_booster(
    params: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    rounds: int,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
) -> lgb.Booster:
    train = lgb.Dataset(_matrix(X), label=y.to_numpy(), free_raw_data=False)
    callbacks: list[Any] = []
    valid_sets = []
    if X_val is not None and y_val is not None:
        valid_sets = [lgb.Dataset(_matrix(X_val), label=y_val.to_numpy(), reference=train)]
        callbacks.append(lgb.early_stopping(EARLY_STOP, verbose=False))
    return lgb.train(
        {**BASE_PARAMS, **params}, train, num_boost_round=rounds, valid_sets=valid_sets, callbacks=callbacks
    )


@dataclass(frozen=True)
class TuneResult:
    params: dict[str, Any]
    best_iteration: int
    val_poisson_deviance: float
    trials: int


def tune(X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, n_trials: int, seed: int) -> TuneResult:
    """Bayesian (TPE) search over LightGBM settings, scored on a later season than it trains on."""
    if n_trials <= 0:
        booster = fit_booster({**DEFAULT_PARAMS, "seed": seed}, X, y, MAX_ROUNDS, X_val, y_val)
        return TuneResult(
            {**DEFAULT_PARAMS, "seed": seed}, booster.best_iteration, float(booster.best_score["valid_0"]["poisson"]), 0
        )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 96, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 400, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 30.0, log=True),
            "seed": seed,
        }
        booster = fit_booster(params, X, y, MAX_ROUNDS, X_val, y_val)
        trial.set_user_attr("best_iteration", booster.best_iteration)
        return float(booster.best_score["valid_0"]["poisson"])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=n_trials)
    best = study.best_trial
    return TuneResult(
        {**best.params, "seed": seed}, int(best.user_attrs["best_iteration"]), float(study.best_value), n_trials
    )


@dataclass
class TargetModel:
    target: str
    model_str: str
    distribution: StatModel
    params: dict[str, Any]
    rounds: int
    _booster: lgb.Booster | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_booster(
        cls, target: str, booster: lgb.Booster, distribution: StatModel, params: dict[str, Any]
    ) -> TargetModel:
        return cls(target, booster.model_to_string(), distribution, params, booster.current_iteration())

    @property
    def dispersion(self) -> Dispersion:
        return self.distribution.dispersion

    @property
    def recalibration(self) -> Recalibration:
        return self.distribution.recalibration

    @property
    def booster(self) -> lgb.Booster:
        if self._booster is None:
            self._booster = lgb.Booster(model_str=self.model_str)
        return self._booster

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.booster.predict(_matrix(X)))

    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        """TreeSHAP values in log space; the last column is the base value."""
        return np.asarray(self.booster.predict(_matrix(X), pred_contrib=True))

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_booster"] = None
        return state


@dataclass
class ModelBundle:
    version: str
    trained_at: str
    models: dict[str, TargetModel]
    copula: np.ndarray
    train_seasons: list[str]
    metrics: dict[str, Any]
    features: list[str] = field(default_factory=lambda: list(FEATURES))

    @property
    def targets(self) -> list[str]:
        return list(self.models)

    @property
    def recalibrations(self) -> list[Recalibration]:
        return [m.recalibration for m in self.models.values()]

    def predict_means(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({t: m.predict(X) for t, m in self.models.items()}, index=X.index)

    def sizes(self, means: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {t: self.models[t].dispersion.size(means[t].to_numpy()) for t in self.models}, index=means.index
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: Path) -> ModelBundle:
        bundle = joblib.load(path)
        if not isinstance(bundle, ModelBundle):
            raise TypeError(f"{path} does not contain a ModelBundle")
        if bundle.features != FEATURES:
            raise ValueError("Model was trained with a different feature set; retrain it")
        return bundle


def explain(model: TargetModel, row: pd.DataFrame, top: int = 8) -> dict[str, Any]:
    """Break one prediction into feature families, in the stat's own units.

    The model works in log space, so SHAP values multiply rather than add. Families are
    applied largest first; each step reports how many points (or rebounds...) it added,
    and the steps sum exactly to the prediction.
    """
    contrib = model.contributions(row)[0]
    base, phi = float(contrib[-1]), contrib[:-1]
    groups: dict[str, float] = {}
    for name, value in zip(FEATURES, phi, strict=True):
        groups[FEATURE_GROUPS[name]] = groups.get(FEATURE_GROUPS[name], 0.0) + float(value)
    running = float(np.exp(base))
    steps = []
    for name, value in sorted(groups.items(), key=lambda kv: -abs(kv[1])):
        new = running * float(np.exp(value))
        steps.append({"group": name, "delta": new - running, "multiplier": float(np.exp(value))})
        running = new
    order = np.argsort(-np.abs(phi))[:top]
    features = [
        {
            "feature": FEATURES[i],
            "value": _clean(row[FEATURES[i]].iloc[0]),
            "effect_pct": 100 * (float(np.exp(phi[i])) - 1),
            "group": FEATURE_GROUPS[FEATURES[i]],
        }
        for i in order
    ]
    return {"base": float(np.exp(base)), "prediction": running, "steps": steps, "top_features": features}


def _clean(value: Any) -> float | None:
    return None if pd.isna(value) else float(value)


def grouped_importance(model: TargetModel) -> dict[str, float]:
    gain = model.booster.feature_importance(importance_type="gain")
    total = float(gain.sum()) or 1.0
    out: dict[str, float] = {}
    for name, g in zip(FEATURES, gain, strict=True):
        out[FEATURE_GROUPS[name]] = out.get(FEATURE_GROUPS[name], 0.0) + float(g) / total
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
