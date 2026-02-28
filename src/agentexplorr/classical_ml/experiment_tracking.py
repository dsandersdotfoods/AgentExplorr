"""
Experiment Tracking with MLflow
=================================

Professional experiment tracking for reproducible machine learning. This module
wraps MLflow to provide a simplified, educational interface for logging
parameters, metrics, artifacts, and models.

WHY TRACK EXPERIMENTS?
    ML development is highly experimental. You try dozens of combinations of:
      - Hyperparameters (learning rate, batch size, layers, etc.)
      - Data preprocessing strategies (normalization, augmentation, feature selection)
      - Model architectures (RF vs XGBoost, CNN vs Transformer)

    Without tracking, you lose track of what you tried and what worked.
    "What learning rate gave the best results last Tuesday?" becomes
    impossible to answer. Experiment tracking is the difference between
    professional ML engineering and chaotic Jupyter notebook exploration.

WHAT MLflow TRACKS:
    1. **Parameters** -- inputs to your experiment (lr=0.001, epochs=10)
    2. **Metrics** -- outputs/results (accuracy=0.95, loss=0.23)
    3. **Artifacts** -- files (model weights, plots, configs)
    4. **Source** -- git commit, code version
    5. **Tags** -- labels for organization (experiment_type=baseline)

THE MLflow WORKFLOW:
    1. Start MLflow server:  `mlflow ui`  (launches at http://localhost:5000)
    2. In your code:
        tracker = ExperimentTracker("my_experiment")
        with tracker.start_run("baseline"):
            tracker.log_params({"lr": 0.001, "model": "random_forest"})
            model.fit(X_train, y_train)
            tracker.log_metrics({"accuracy": 0.95, "f1": 0.93})
            tracker.log_model(model)
    3. View results in the MLflow UI: compare runs side-by-side

EXPERIMENT ORGANIZATION:
    MLflow uses a hierarchy:
      Experiment (e.g., "wine_classification")
        +-- Run (e.g., "random_forest_v1")
        |     +-- Parameters: {n_estimators: 100, max_depth: 10}
        |     +-- Metrics: {accuracy: 0.95, f1: 0.93}
        |     +-- Artifacts: model.pkl, confusion_matrix.png
        +-- Run (e.g., "random_forest_v2_tuned")
              +-- Parameters: {n_estimators: 200, max_depth: 20}
              +-- Metrics: {accuracy: 0.97, f1: 0.96}

    This makes it easy to compare runs and find the best model.

MODEL REGISTRY:
    MLflow's Model Registry is like version control for models:
      - Register a model with a name (e.g., "wine_classifier")
      - Track versions (v1, v2, v3...)
      - Transition between stages: "Staging" -> "Production" -> "Archived"
      - Roll back to previous versions if a new model underperforms

LEARNING RESOURCES:
    - MLflow Docs: https://mlflow.org/docs/latest/index.html
    - MLflow Quickstart: https://mlflow.org/docs/latest/getting-started/intro-quickstart/index.html
    - MLflow Tracking: https://mlflow.org/docs/latest/tracking.html
    - MLflow Model Registry: https://mlflow.org/docs/latest/model-registry.html
    - VIDEO: "MLflow Tutorial" -- https://www.youtube.com/watch?v=ksYIVDue8ak
    - VIDEO: "MLflow End-to-End" -- https://www.youtube.com/watch?v=859OxXrt_TI
    - VIDEO: "Experiment Tracking Best Practices" -- https://www.youtube.com/watch?v=dPmH3G9NQtY
    - Weights & Biases (alternative): https://wandb.ai/
    - DVC (Data Version Control): https://dvc.org/
"""

from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

from agentexplorr.core.config import get_settings
from agentexplorr.core.logging import get_logger

logger = get_logger(__name__)


class ExperimentTracker:
    """Wrapper around MLflow for experiment tracking.

    Provides a simplified, educational interface for the most common MLflow
    operations: logging parameters, metrics, artifacts, and models. Includes
    model registry support and cross-run comparison.

    The class uses lazy loading for MLflow -- it won't be imported until
    you actually call a method that needs it. This means you can import
    ExperimentTracker even if MLflow isn't installed (useful for testing).

    Attributes:
        experiment_name: Name of the MLflow experiment.
        tracking_uri: URI of the MLflow tracking server.

    Example:
        >>> tracker = ExperimentTracker(experiment_name="classification")
        >>>
        >>> with tracker.start_run(run_name="random_forest_v1"):
        ...     tracker.log_params({"n_estimators": 100, "max_depth": 10})
        ...     tracker.log_metrics({"accuracy": 0.95, "f1": 0.93})
        ...     tracker.log_artifact("model.pkl")

    Or use the context manager with the run ID:
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
            experiment_name: MLflow experiment name. Groups related runs together.
                If None, uses the default from settings (Settings.mlflow_experiment_name).
            tracking_uri: MLflow server URI. Can be:
                - "http://localhost:5000" (local MLflow server)
                - "sqlite:///mlruns.db" (local database)
                - "./mlruns" (local file system, simplest option)
                If None, uses the default from settings.
        """
        settings = get_settings()
        self.experiment_name: str = experiment_name or settings.mlflow_experiment_name
        self.tracking_uri: str = tracking_uri or settings.mlflow_tracking_uri
        self._active_run_id: str | None = None
        self._mlflow: Any = None

        logger.info(
            "experiment_tracker_initialized",
            experiment_name=self.experiment_name,
            tracking_uri=self.tracking_uri,
        )

    def _get_mlflow(self) -> Any:
        """Lazy-load MLflow to avoid import errors when not installed.

        WHY LAZY LOADING?
            Not every user will have MLflow installed. By lazy-loading,
            the rest of the module works fine without MLflow. The error
            only appears when you actually try to use tracking features.

        Returns:
            The mlflow module.

        Raises:
            ImportError: If MLflow is not installed.
        """
        if self._mlflow is None:
            try:
                import mlflow

                mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(self.experiment_name)
                self._mlflow = mlflow
                logger.info(
                    "mlflow_initialized",
                    tracking_uri=self.tracking_uri,
                    experiment_name=self.experiment_name,
                )
            except ImportError:
                msg = (
                    "MLflow not installed. Install it with:\n"
                    "  pip install mlflow\n"
                    "Or, if using the project's extras:\n"
                    "  uv sync --extra ml"
                )
                raise ImportError(msg)  # noqa: B904
        return self._mlflow

    @contextmanager
    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        description: str | None = None,
    ) -> Generator[str, None, None]:
        """Start an MLflow run as a context manager.

        Everything logged between entering and exiting the context is grouped
        into a single run. The run is automatically ended on exit (even if
        an exception occurs).

        HOW RUNS WORK:
            A "run" is a single execution of your ML pipeline. Each run
            records all parameters, metrics, and artifacts together.
            You can compare runs in the MLflow UI to see which configuration
            performed best.

        Args:
            run_name: Human-readable name for the run (e.g., "rf_baseline_v1").
                This appears in the MLflow UI. If None, MLflow auto-generates
                a random name like "dazzling-panda-42".
            tags: Optional key-value pairs for categorization.
                Examples: {"model_type": "random_forest", "dataset": "wine"}
                Tags are searchable in the MLflow UI.
            description: Optional description of what this run is testing.

        Yields:
            The MLflow run ID (a unique identifier for this run).

        Example:
            >>> with tracker.start_run("my_experiment", tags={"type": "baseline"}):
            ...     tracker.log_params({"lr": 0.001})
            ...     tracker.log_metrics({"accuracy": 0.95})
        """
        mlflow = self._get_mlflow()

        # Merge description into tags if provided
        if description and tags is None:
            tags = {"description": description}
        elif description and tags is not None:
            tags["description"] = description

        with mlflow.start_run(run_name=run_name, tags=tags) as run:
            self._active_run_id = run.info.run_id
            logger.info(
                "mlflow_run_started",
                run_id=self._active_run_id,
                run_name=run_name,
                experiment=self.experiment_name,
            )
            try:
                yield self._active_run_id
            finally:
                self._active_run_id = None
                logger.info("mlflow_run_ended", run_id=run.info.run_id)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters (inputs to the experiment).

        Parameters are things you SET BEFORE training:
          - learning_rate, batch_size, epochs
          - model_type, n_estimators, max_depth
          - preprocessing steps, feature selection criteria

        Parameters are logged ONCE per run (they don't change over time).
        They appear as columns in the MLflow UI's comparison table.

        IMPORTANT: Parameter values are stored as strings with a 500-char limit.
        For complex values, consider logging them as artifacts (log_dict).

        Args:
            params: Dictionary of parameter names and values.
                Values are automatically converted to strings.

        Example:
            >>> tracker.log_params({
            ...     "model": "RandomForest",
            ...     "n_estimators": 100,
            ...     "max_depth": 10,
            ...     "learning_rate": 0.001,
            ... })
        """
        mlflow = self._get_mlflow()
        # MLflow requires string values, and has a limit on the number of
        # params logged at once (100). We handle this by converting and
        # logging in batches.
        str_params = {k: str(v) for k, v in params.items()}
        mlflow.log_params(str_params)
        logger.info("params_logged", n_params=len(params))

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """Log metrics (outputs/results of the experiment).

        Metrics are things you MEASURE during or after training:
          - loss, accuracy, precision, recall, F1 score
          - MSE, RMSE, MAE, R-squared
          - silhouette score, inertia

        Metrics can be logged at different steps to track progress over time
        (e.g., loss at each epoch). The MLflow UI can plot these as time series.

        Args:
            metrics: Dictionary of metric names and values.
            step: Optional step number. Use this for epoch-level metrics:
                step=1 for epoch 1, step=2 for epoch 2, etc.
                This enables time-series visualization in MLflow UI.

        Example:
            >>> # Log final metrics (no step)
            >>> tracker.log_metrics({"test_accuracy": 0.95, "test_f1": 0.93})
            >>>
            >>> # Log per-epoch metrics
            >>> for epoch in range(10):
            ...     loss = train_one_epoch()
            ...     tracker.log_metrics({"train_loss": loss}, step=epoch)
        """
        mlflow = self._get_mlflow()
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str) -> None:
        """Log a file as an artifact.

        Artifacts are FILES produced by the experiment:
          - Model weights (.pt, .pkl, .joblib)
          - Plots and visualizations (.png, .html)
          - Predictions (.csv)
          - Configuration files (.yaml, .json)
          - Data samples or summaries

        Artifacts are stored in MLflow's artifact store (local filesystem
        or cloud storage like S3) and can be downloaded later.

        Args:
            path: Path to the file to log.

        Example:
            >>> # Save a plot and log it
            >>> plt.savefig("confusion_matrix.png")
            >>> tracker.log_artifact("confusion_matrix.png")
        """
        mlflow = self._get_mlflow()
        mlflow.log_artifact(path)
        logger.info("artifact_logged", path=path)

    def log_dict(
        self,
        data: dict[str, Any],
        filename: str = "data.json",
    ) -> None:
        """Log a dictionary as a JSON artifact.

        Useful for logging complex data that doesn't fit in parameters:
          - Full hyperparameter grids
          - Feature importance dictionaries
          - Classification reports
          - Configuration objects

        Args:
            data: Dictionary to log (must be JSON-serializable).
            filename: Name for the artifact file.

        Example:
            >>> tracker.log_dict(
            ...     {"feature_importances": {"alcohol": 0.3, "color": 0.2}},
            ...     filename="feature_importances.json",
            ... )
        """
        mlflow = self._get_mlflow()

        # Write dict to a temporary JSON file, then log as artifact
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f, indent=2, default=str)
            temp_path = f.name

        mlflow.log_artifact(temp_path, artifact_path="dicts")
        # Clean up temp file
        Path(temp_path).unlink(missing_ok=True)

        logger.info("dict_logged", filename=filename, keys=list(data.keys()))

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: str | None = None,
    ) -> None:
        """Log a model to MLflow, with optional Model Registry registration.

        MLflow automatically detects the framework (sklearn, PyTorch, etc.)
        and uses the appropriate serialization method.

        MODEL REGISTRY:
            If registered_model_name is provided, the model is also registered
            in MLflow's Model Registry. This provides:
              - Version tracking (v1, v2, v3...)
              - Stage management (Staging -> Production -> Archived)
              - Model lineage (which run produced this model?)
              - Easy deployment (load by name instead of path)

        Args:
            model: The trained model (sklearn, PyTorch, etc.).
            artifact_path: Path within the run's artifact store.
            registered_model_name: If provided, register the model with
                this name in MLflow's Model Registry. Use this for models
                you want to track across versions and deploy.

        Example:
            >>> # Log without registration
            >>> tracker.log_model(trained_pipeline)
            >>>
            >>> # Log with registration (for deployment tracking)
            >>> tracker.log_model(
            ...     trained_pipeline,
            ...     registered_model_name="wine_classifier",
            ... )
        """
        mlflow = self._get_mlflow()

        # Detect model framework and use appropriate logger
        model_type = "unknown"

        try:
            import sklearn.base

            if isinstance(model, sklearn.base.BaseEstimator):
                mlflow.sklearn.log_model(
                    model,
                    artifact_path,
                    registered_model_name=registered_model_name,
                )
                model_type = "sklearn"
                logger.info(
                    "model_logged",
                    framework="sklearn",
                    artifact_path=artifact_path,
                    registered_name=registered_model_name,
                )
                return
        except ImportError:
            pass

        try:
            import torch.nn

            if isinstance(model, torch.nn.Module):
                mlflow.pytorch.log_model(
                    model,
                    artifact_path,
                    registered_model_name=registered_model_name,
                )
                model_type = "pytorch"
                logger.info(
                    "model_logged",
                    framework="pytorch",
                    artifact_path=artifact_path,
                    registered_name=registered_model_name,
                )
                return
        except ImportError:
            pass

        # Fallback: warn the user
        logger.warning(
            "unknown_model_type",
            model_type=type(model).__name__,
            hint="Model was not logged. Supported frameworks: sklearn, pytorch.",
        )

    def register_model(
        self,
        model_uri: str,
        name: str,
    ) -> Any:
        """Register a logged model in MLflow's Model Registry.

        The Model Registry provides centralized model management:
          - Version control for models (v1, v2, v3...)
          - Stage transitions (None -> Staging -> Production -> Archived)
          - Annotations and descriptions
          - Lineage tracking (which experiment produced this model?)

        Args:
            model_uri: URI of the logged model (e.g., "runs:/<run_id>/model").
            name: Name to register the model under.

        Returns:
            The registered ModelVersion object.

        Example:
            >>> with tracker.start_run("train_model") as run_id:
            ...     tracker.log_model(model)
            ...     model_version = tracker.register_model(
            ...         f"runs:/{run_id}/model",
            ...         name="wine_classifier",
            ...     )
            ...     print(f"Registered v{model_version.version}")
        """
        mlflow = self._get_mlflow()
        model_version = mlflow.register_model(model_uri, name)
        logger.info(
            "model_registered",
            name=name,
            version=model_version.version,
        )
        return model_version

    def compare_runs(
        self,
        metric_key: str = "accuracy",
        n_top: int = 5,
    ) -> list[dict[str, Any]]:
        """Compare runs in the current experiment, sorted by a metric.

        This is useful for finding the best-performing run across all your
        experiments. It queries the MLflow tracking server for all runs in
        the current experiment and returns them sorted by the specified metric.

        Args:
            metric_key: Which metric to sort by (e.g., "accuracy", "f1_score").
            n_top: Number of top runs to return.

        Returns:
            List of dictionaries, each containing run info (run_id, params, metrics).
            Sorted by the specified metric in descending order.

        Example:
            >>> results = tracker.compare_runs(metric_key="accuracy", n_top=3)
            >>> for r in results:
            ...     print(f"Run {r['run_id'][:8]}... Acc={r['metrics']['accuracy']:.4f}")
        """
        mlflow = self._get_mlflow()

        # Search for all runs in the experiment, sorted by the metric
        runs = mlflow.search_runs(
            experiment_names=[self.experiment_name],
            order_by=[f"metrics.{metric_key} DESC"],
            max_results=n_top,
        )

        if runs.empty:
            logger.warning("no_runs_found", experiment=self.experiment_name)
            return []

        comparison: list[dict[str, Any]] = []
        for _, row in runs.iterrows():
            run_info: dict[str, Any] = {
                "run_id": row.get("run_id", ""),
                "run_name": row.get("tags.mlflow.runName", ""),
                "status": row.get("status", ""),
                "start_time": str(row.get("start_time", "")),
                "metrics": {},
                "params": {},
            }

            # Extract metrics (columns starting with "metrics.")
            for col in runs.columns:
                if col.startswith("metrics."):
                    metric_name = col.replace("metrics.", "")
                    value = row[col]
                    if value is not None and str(value) != "nan":
                        run_info["metrics"][metric_name] = float(value)

            # Extract params (columns starting with "params.")
            for col in runs.columns:
                if col.startswith("params."):
                    param_name = col.replace("params.", "")
                    value = row[col]
                    if value is not None and str(value) != "nan":
                        run_info["params"][param_name] = str(value)

            comparison.append(run_info)

        logger.info(
            "runs_compared",
            experiment=self.experiment_name,
            metric=metric_key,
            n_runs=len(comparison),
        )

        # Pretty-print the comparison
        print("\n" + "=" * 70)
        print(f"TOP {n_top} RUNS (sorted by {metric_key})")
        print(f"Experiment: {self.experiment_name}")
        print("=" * 70)
        for i, run in enumerate(comparison, 1):
            print(f"\n  #{i}: {run['run_name'] or run['run_id'][:8]}")
            if run["metrics"]:
                metrics_str = ", ".join(
                    f"{k}={v:.4f}" for k, v in sorted(run["metrics"].items())
                )
                print(f"      Metrics: {metrics_str}")
            if run["params"]:
                params_str = ", ".join(
                    f"{k}={v}" for k, v in sorted(run["params"].items())
                )
                print(f"      Params:  {params_str}")
        print("=" * 70)

        return comparison

    def get_best_run(self, metric_key: str = "accuracy") -> dict[str, Any] | None:
        """Get the single best run for a given metric.

        Convenience method that wraps compare_runs(n_top=1).

        Args:
            metric_key: Which metric to optimize.

        Returns:
            Dictionary with the best run's info, or None if no runs exist.
        """
        results = self.compare_runs(metric_key=metric_key, n_top=1)
        return results[0] if results else None
