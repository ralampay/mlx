"""Default adapter infrastructure; alternative backends are injected by Python callers."""


def create_experiment_backend():
    from .libreyolo.adapter_execution import LibreYOLOExperimentBackend
    return LibreYOLOExperimentBackend()
