"""Public detection exports, loaded only when requested."""
from importlib import import_module

_EXPORTS = {
    'BoundingBox': ('mlx.modes.object_detection.tracking.models', 'BoundingBox'),
    'CLASS_AWARE_SCHEMA_VERSION': ('mlx.modes.object_detection.tracking.class_aware', 'CLASS_AWARE_SCHEMA_VERSION'),
    'ClassAwareTrackingRecord': ('mlx.modes.object_detection.tracking.class_aware', 'ClassAwareTrackingRecord'),
    'ClassAwareTrackingResultWriter': ('mlx.modes.object_detection.tracking.class_aware', 'ClassAwareTrackingResultWriter'),
    'CreateTrackingAlgorithm': ('mlx.modes.object_detection.tracking.registry', 'CreateTrackingAlgorithm'),
    'ExportTrackingReplay': ('mlx.modes.object_detection.tracking.replay', 'ExportTrackingReplay'),
    'ExportMOTFromClassAwareTracking': ('mlx.modes.object_detection.tracking.class_aware', 'ExportMOTFromClassAwareTracking'),
    'MOTRecord': ('mlx.modes.object_detection.tracking.mot', 'MOTRecord'),
    'MOTExportResult': ('mlx.modes.object_detection.tracking.class_aware', 'MOTExportResult'),
    'RunObjectDetectionTrackingCommand': ('mlx.modes.object_detection.tracking.detection', 'RunObjectDetectionTrackingCommand'),
    'RunTrackByDetectionCommand': ('mlx.modes.object_detection.tracking.command', 'RunTrackByDetectionCommand'),
    'RunTrackingVideo': ('mlx.modes.object_detection.tracking.session', 'RunTrackingVideo'),
    'TrackResult': ('mlx.modes.object_detection.tracking.models', 'TrackResult'),
    'TrackStatus': ('mlx.modes.object_detection.tracking.models', 'TrackStatus'),
    'TrackingAlgorithm': ('mlx.modes.object_detection.tracking.protocols', 'TrackingAlgorithm'),
    'TrackingBenchmarkResult': ('mlx.modes.object_detection.tracking.evaluation', 'TrackingBenchmarkResult'),
    'TrackingDetection': ('mlx.modes.object_detection.tracking.models', 'TrackingDetection'),
    'TrackingFrameResult': ('mlx.modes.object_detection.tracking.models', 'TrackingFrameResult'),
    'TrackingRequest': ('mlx.modes.object_detection.tracking.requests', 'TrackingRequest'),
    'TrackingReplayResult': ('mlx.modes.object_detection.tracking.replay', 'TrackingReplayResult'),
    'TrackingResultWriter': ('mlx.modes.object_detection.tracking.class_aware', 'TrackingResultWriter'),
    'TrackingRunResult': ('mlx.modes.object_detection.tracking.session', 'TrackingRunResult'),
    'TrackerRegistry': ('mlx.modes.object_detection.tracking.registry', 'TrackerRegistry'),
    'list_trackers': ('mlx.modes.object_detection.tracking.registry', 'list_trackers'),
    'register_tracker': ('mlx.modes.object_detection.tracking.registry', 'register_tracker'),
    'read_class_aware_tracking_file': ('mlx.modes.object_detection.tracking.class_aware', 'read_class_aware_tracking_file'),
    'to_tracking_detections': ('mlx.modes.object_detection.tracking.detection', 'to_tracking_detections'),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    return getattr(import_module(module), attribute)


def __dir__():
    return sorted(set(globals()) | set(__all__))
