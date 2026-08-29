from typing import Protocol

from mlx.core.aws.commands import (
    AwsTrainingService,
    GetAwsTrainingStatus,
    ResumeAwsTraining,
    StopAwsTraining,
    SubmitAwsTraining,
    WatchAwsTraining,
)
from mlx.modes.object_detection.aws.models import AwsBestModelLocation


class AwsBestModelLocator(Protocol):
    def locate_best_model(self) -> AwsBestModelLocation: ...


class LocateBestAwsObjectDetectionModel:
    def __init__(self, locator: AwsBestModelLocator) -> None:
        self.locator = locator

    def execute(self) -> AwsBestModelLocation:
        return self.locator.locate_best_model()


class SubmitAwsObjectDetectionTraining(SubmitAwsTraining):
    pass


class ResumeAwsObjectDetectionTraining(ResumeAwsTraining):
    pass


class GetAwsObjectDetectionTrainingStatus(GetAwsTrainingStatus):
    pass


class WatchAwsObjectDetectionTraining(WatchAwsTraining):
    pass


class StopAwsObjectDetectionTraining(StopAwsTraining):
    pass


__all__ = [
    "AwsBestModelLocator",
    "AwsTrainingService",
    "GetAwsObjectDetectionTrainingStatus",
    "LocateBestAwsObjectDetectionModel",
    "ResumeAwsObjectDetectionTraining",
    "StopAwsObjectDetectionTraining",
    "SubmitAwsObjectDetectionTraining",
    "WatchAwsObjectDetectionTraining",
]
