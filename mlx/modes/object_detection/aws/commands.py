from mlx.core.aws.commands import (
    AwsTrainingService,
    GetAwsTrainingStatus,
    ResumeAwsTraining,
    StopAwsTraining,
    SubmitAwsTraining,
    WatchAwsTraining,
)


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
    "AwsTrainingService",
    "GetAwsObjectDetectionTrainingStatus",
    "ResumeAwsObjectDetectionTraining",
    "StopAwsObjectDetectionTraining",
    "SubmitAwsObjectDetectionTraining",
    "WatchAwsObjectDetectionTraining",
]
