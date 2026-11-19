"""Runtime configuration, overridable with NBA_* environment variables or a .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NBA_", env_file=".env", extra="ignore")

    data_dir: Path = ROOT / "data"
    artifacts_dir: Path = ROOT / "models"

    # Walk forward evaluation: fold A trains on season 1 and validates on season 2 (used for tuning),
    # fold B trains on seasons 1-2 and tests on season 3, which nothing in training ever touched.
    train_seasons: list[str] = Field(default_factory=lambda: ["2023-24", "2024-25"])
    test_season: str = "2025-26"

    optuna_trials: int = 20
    random_state: int = 42
    simulations: int = 10_000

    request_timeout: int = 120
    retry_attempts: int = 4  # first try plus 3 retries
    retry_wait_seconds: float = 30.0
    live_refresh_hours: float = 6.0

    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    mlflow_tracking_uri: str = f"sqlite:///{(ROOT / 'mlruns' / 'mlflow.db').as_posix()}"

    @property
    def bundle_path(self) -> Path:
        return self.artifacts_dir / "nba_predictor.joblib"

    @property
    def report_path(self) -> Path:
        return self.artifacts_dir / "report.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
