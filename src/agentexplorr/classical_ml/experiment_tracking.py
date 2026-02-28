"""
Experiment Tracking with MLflow
=================================

WHY TRACK EXPERIMENTS?
  ML development is highly experimental. You try dozens of combinations of:
  - Hyperparameters (learning rate, batch size, layers, etc.)
  - Data preprocessing strategies
  - Model architectures

  Without tracking, you lose track of what you tried and what worked.
  "What learning rate gave the best results last Tuesday?" becomes
  impossible to answer.

WHAT MLflow TRACKS:
  1. **Parameters** — inputs to your experiment (lr=0.001, epochs=10)
  2. **Metrics** — outputs/results (accuracy=0.95, loss=0.23)
  3. **Artifacts** — files (model weights, plots, configs)
  4. **Source** — git commit, code version
  5. **Tags** — labels for organization (experiment_type=baseline)

THE MLflow UI:
  Run `mlflow ui` to launch a web dashboard at http://localhost:5000
  where you can compare experiments, visualize metrics over time,
  and find the best performing model.

LEARNING RESOURCES:
  - MLflow docs: https://mlflow.org/docs/latest/index.html
  - VIDEO: "MLflow Tutorial" — https://www.youtube.com/watch?v=ksYIVDue8ak
  - VIDEO: "Experiment Tracking Best Practices" — https://www.youtube.com/watch?v=dPmH3G9NQtY
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

from agentexplorr.core.config import get_settings
from agentexplorr.core.logging import get_logger

logger = get_logger(__name__)


class ExperimentTracker:
    """Wrapper around MLflow for experiment tracking.

    Provides a simplified interface for the most common MLflow operations.
    Handles MLflow initialization, run management, and error handling.

    Example:
        >>> tracker = ExperimentTracker(experiment_name="classification")
        >>>
        >>> with tracker.start_run(run_name="random_forest_v1"):
        ...     tracker.log_params({"n_estimators": 100, "max_depth": 10})
        ...     tracker.log_metrics({"accuracy": 0.95, "f1": 0.93})
        ...     tracker.log_artifact("model.pkl")

    Or use the context manager directly:
        >>> with tracker.start_run("my_run") as run_id:
        ...     tracker.log_metrics({"loss": 0.5})
        ...     print(f"Run ID: {run_id}")
    """

    def __init__(
        self,
        experiment_name: str | None = None,
        tracking_uri: str | None = None,
    ) -> None:
        """Initialize the experiment tracker.

        Args:
            experiment_name: MLflow experiment name. Defaults to settings.
            tracking_uri: MLflow server URI. Defaults to settings.
        """
        settings = get_settings()
        self.experiment_name = experiment_name or settings.mlflow_experiment_name
        self.tracking_uri = tracking_uri or settings.mlflow_tracking_uri
        self._active_run_id: str | None = None
        self._mlflow: Any = None

    def _get_mlflow(self) -> Any:
        """Lazy-load MLflow to avoid import errors when not installed."""
        if self._mlflow is None:
            try:
                import mlflow

                mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(self.experiment_name)
                self._mlflow = mlflow
            except ImportError:
                msg = "MLflow not installed. Run: uv sync --extra ml"
                raise ImportError(msg)  # noqa: B904
        return self._mlflow

    @contextmanager
    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> Generator[str, None, None]:
        """Start an MLflow run as a context manager.

        Everything logged between __enter__ and __exit__ is grouped
        into a single run. The run is automatically ended on exit.

        Args:
            run_name: Human-readable name for the run.
            tags: Optional tags for categorization.

        Yields:
            The MLflow run ID.
        """
        mlflow = self._get_mlflow()

        with mlflow.start_run(run_name=run_name, tags=tags) as run:
            self._active_run_id = run.info.run_id
            logger.info(
                "mlflow_run_started",
                run_id=self._active_run_id,
                run_name=run_name,
            )
            try:
                yield self._active_run_id
            finally:
                self._active_run_id = None
                logger.info("mlflow_run_ended", run_id=run.info.run_id)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters (inputs to the experiment).

        Parameters are things you SET before training:
        learning_rate, batch_size, model_type, etc.

        Args:
            params: Dict of parameter names and values.
        """
        mlflow = self._get_mlflow()
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics (outputs/results of the experiment).

        Metrics are things you MEASURE during or after training:
        loss, accuracy, F1 score, etc.

        Args:
            metrics: Dict of metric names and values.
            step: Optional step number (for time-series metrics).
        """
        mlflow = self._get_mlflow()
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str) -> None:
        """Log a file as an artifact.

        Artifacts are files produced by the experiment:
        model weights, plots, predictions, configs, etc.

        Args:
            path: Path to the file to log.
        """
        mlflow = self._get_mlflow()
        mlflow.log_artifact(path)

    def log_model(self, model: Any, artifact_path: str = "model") -> None:
        """Log a model to MLflow's model registry.

        Args:
            model: The trained model (sklearn, PyTorch, etc.).
            artifact_path: Path within the run's artifacts.
        """
        mlflow = self._get_mlflow()

        # Detect model framework and use appropriate logger
        try:
            import sklearn.base

            if isinstance(model, sklearn.base.BaseEstimator):
                mlflow.sklearn.log_model(model, artifact_path)
                return
        except ImportError:
            pass

        try:
            import torch.nn

            if isinstance(model, torch.nn.Module):
                mlflow.pytorch.log_model(model, artifact_path)
                return
        except ImportError:
            pass

        # Fallback: log as a generic artifact
        logger.warning("unknown_model_type", model_type=type(model).__name__)
