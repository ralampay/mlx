"""Public detection exports, loaded only when requested."""
from importlib import import_module

_EXPORTS = {
    'BoundingBox': ('mlx.modes.object_detection.tracking', 'BoundingBox'),
    'BenchmarkObjectDetectionModel': ('mlx.modes.object_detection.commands', 'BenchmarkObjectDetectionModel'),
    'BenchmarkObjectDetectionRequest': ('mlx.modes.object_detection.requests', 'BenchmarkObjectDetectionRequest'),
    'ClassAwareTrackingRecord': ('mlx.modes.object_detection.tracking', 'ClassAwareTrackingRecord'),
    'ConvertObjectDetectionModel': ('mlx.modes.object_detection.commands', 'ConvertObjectDetectionModel'),
    'CreateObjectDetector': ('mlx.modes.object_detection.commands', 'CreateObjectDetector'),
    'Detection': ('mlx.modes.object_detection.models', 'Detection'),
    'DetectionAdapter': ('mlx.modes.object_detection.models', 'DetectionAdapter'),
    'DetectionResult': ('mlx.modes.object_detection.models', 'DetectionResult'),
    'ExportTrackingReplay': ('mlx.modes.object_detection.tracking', 'ExportTrackingReplay'),
    'ExportMOTFromClassAwareTracking': ('mlx.modes.object_detection.tracking', 'ExportMOTFromClassAwareTracking'),
    'FineTuneObjectDetectionModel': ('mlx.modes.object_detection.commands', 'FineTuneObjectDetectionModel'),
    'FineTuneObjectDetectionRequest': ('mlx.modes.object_detection.requests', 'FineTuneObjectDetectionRequest'),
    'ListObjectDetectionModels': ('mlx.modes.object_detection.commands', 'ListObjectDetectionModels'),
    'ObjectDetectionBenchmarkResult': ('mlx.modes.object_detection.models', 'ObjectDetectionBenchmarkResult'),
    'RunObjectDetectionStream': ('mlx.modes.object_detection.commands', 'RunObjectDetectionStream'),
    'RunObjectDetectionTrackingCommand': ('mlx.modes.object_detection.tracking', 'RunObjectDetectionTrackingCommand'),
    'RunTrackByDetectionCommand': ('mlx.modes.object_detection.tracking', 'RunTrackByDetectionCommand'),
    'TrackResult': ('mlx.modes.object_detection.tracking', 'TrackResult'),
    'TrackStatus': ('mlx.modes.object_detection.tracking', 'TrackStatus'),
    'TrackingAlgorithm': ('mlx.modes.object_detection.tracking', 'TrackingAlgorithm'),
    'TrackingDetection': ('mlx.modes.object_detection.tracking', 'TrackingDetection'),
    'TrackingFrameResult': ('mlx.modes.object_detection.tracking', 'TrackingFrameResult'),
    'TrackingRequest': ('mlx.modes.object_detection.tracking', 'TrackingRequest'),
    'TrackingReplayResult': ('mlx.modes.object_detection.tracking', 'TrackingReplayResult'),
    'TrackingRunResult': ('mlx.modes.object_detection.tracking', 'TrackingRunResult'),
    'RunTrackingVideo': ('mlx.modes.object_detection.tracking', 'RunTrackingVideo'),
    'TrainObjectDetectionModel': ('mlx.modes.object_detection.commands', 'TrainObjectDetectionModel'),
    'to_tracking_detections': ('mlx.modes.object_detection.tracking', 'to_tracking_detections'),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    return getattr(import_module(module), attribute)


def __dir__():
    return sorted(set(globals()) | set(__all__))
