"""Default adapter infrastructure; alternative backends are injected by Python callers."""


def create_experiment_backend():
    from .libreyolo.adapter_execution import LibreYOLOExperimentBackend
    return LibreYOLOExperimentBackend()


ADAPTER_ACTIONS = frozenset({
    "adapter-verify", "adapter-prepare", "adapter-calibrate", "adapter-experiment",
    "adapter-report", "adapter-slice-predict", "adapter-slice-report",
})


def run_adapter_action(config):
    """Construct only the inputs and dependencies needed by the selected action."""
    from mlx.core.exceptions import MLXUserError
    from .adapter_configuration import adapter_path

    action = config.get("action")
    if action not in ADAPTER_ACTIONS:
        raise MLXUserError(f"Unsupported adapter action: {action!r}")
    if action == "adapter-verify":
        from .libreyolo.adapter_backend import VerifyFoundationCheckpoint
        _, info = VerifyFoundationCheckpoint(
            config.get("model") or "yolox-l",
            adapter_path(config, "model_path", "--checkpoint"),
        ).execute()
        return info
    if action == "adapter-prepare":
        from .adapter_data import PrepareDawnAdapterDataset
        return PrepareDawnAdapterDataset(
            adapter_path(config, "dataset_path", "--dataset"),
            adapter_path(config, "output_path", "--output"),
            seed=config.get("random_seed") if config.get("random_seed") is not None else 42,
        ).execute()
    if action == "adapter-report":
        from .adapter_report import GenerateAdapterReport
        return GenerateAdapterReport(
            adapter_path(config, "output_path", "--output"),
            baseline_study=config.get("baseline_study"),
            comparison_method=config.get("comparison_method") or "drax",
        ).execute()
    if action in {"adapter-slice-predict", "adapter-slice-report"}:
        from .adapter_slices import AdapterSliceRequest, CacheAdapterSlicePredictions, GenerateAdapterSliceReport
        command = CacheAdapterSlicePredictions if action == "adapter-slice-predict" else GenerateAdapterSliceReport
        return command(AdapterSliceRequest.from_config(config)).execute()
    from .adapter_experiment import AdapterExperimentRequest, RunAdapterExperiment
    request = AdapterExperimentRequest.from_config(config)
    if action == "adapter-calibrate":
        from .libreyolo.adapter_execution import CalibrateAdapterExperiment
        return CalibrateAdapterExperiment(request).execute()
    return RunAdapterExperiment(request, backend=create_experiment_backend()).execute()
